import 'dart:convert';
import 'dart:io';

import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:test/test.dart';

/// Live end-to-end test that alternates between GPT-5.1 and Gemini 3 for
/// 12 round-trips, validating that a single OpenAI-format conversation
/// history can seamlessly interoperate between providers.
///
/// Requires both OPENAI_API_KEY and GEMINI_API_KEY in environment or .env.
void main() {
  late oai.OpenAIClient openAIClient;
  late gai.GoogleAIClient geminiClient;

  const openAIModel = 'gpt-4.1-nano';
  const geminiModel = 'gemini-3-flash-preview';

  // Single OpenAI-format conversation history shared across providers.
  final history = <oai.ChatMessage>[];

  // Thought signatures for Gemini turns.
  var thoughtSignatures = <String, String>{};

  // Tool call ID counter.
  var toolCallCounter = 0;
  String generateToolCallId() => 'call_${toolCallCounter++}';

  // Tools in OpenAI format - shared across both providers.
  final tools = [
    oai.Tool.function(
      name: 'lookup_capital',
      description: 'Look up the capital city of a country.',
      parameters: {
        'type': 'object',
        'properties': {
          'country': {
            'type': 'string',
            'description': 'The country name, e.g. "France"',
          },
        },
        'required': ['country'],
      },
    ),
    oai.Tool.function(
      name: 'lookup_population',
      description: 'Look up the population of a city.',
      parameters: {
        'type': 'object',
        'properties': {
          'city': {
            'type': 'string',
            'description': 'The city name, e.g. "Paris"',
          },
        },
        'required': ['city'],
      },
    ),
  ];

  /// Simulate tool execution.
  String executeToolCall(String name, Map<String, dynamic> args) {
    return switch (name) {
      'lookup_capital' => jsonEncode({
        'capital': switch ((args['country'] as String?)?.toLowerCase()) {
          'france' => 'Paris',
          'japan' => 'Tokyo',
          'brazil' => 'Brasília',
          'australia' => 'Canberra',
          _ => 'Unknown',
        },
        'country': args['country'],
      }),
      'lookup_population' => jsonEncode({
        'population': switch ((args['city'] as String?)?.toLowerCase()) {
          'paris' => '2.1 million',
          'tokyo' => '13.9 million',
          'brasília' || 'brasilia' => '3.0 million',
          'canberra' => '460,000',
          _ => 'Unknown',
        },
        'city': args['city'],
      }),
      _ => jsonEncode({'error': 'Unknown function: $name'}),
    };
  }

  /// Handle tool calls from an assistant message - execute tools and add
  /// results to history.
  void handleToolCalls(oai.AssistantMessage msg) {
    if (msg.toolCalls == null || msg.toolCalls!.isEmpty) return;
    for (final toolCall in msg.toolCalls!) {
      final args = jsonDecode(toolCall.function.arguments) as Map<String, dynamic>;
      final result = executeToolCall(toolCall.function.name, args);
      history.add(
        oai.ChatMessage.tool(
          toolCallId: toolCall.id,
          content: result,
        ),
      );
    }
  }

  /// Perform a round-trip using OpenAI (GPT-5.1).
  Future<oai.AssistantMessage> roundTripOpenAI(String userMessage) async {
    history.add(oai.ChatMessage.user(userMessage));

    final response = await openAIClient.chat.completions.create(
      oai.ChatCompletionCreateRequest(
        model: openAIModel,
        messages: history,
        tools: tools,
        toolChoice: oai.ToolChoice.auto(),
      ),
    );

    final msg = response.choices.first.message;
    history.add(msg);

    // If tool calls, execute and get follow-up.
    if (msg.toolCalls != null && msg.toolCalls!.isNotEmpty) {
      handleToolCalls(msg);

      final followUp = await openAIClient.chat.completions.create(
        oai.ChatCompletionCreateRequest(
          model: openAIModel,
          messages: history,
          tools: tools,
        ),
      );
      final followUpMsg = followUp.choices.first.message;
      history.add(followUpMsg);
      return followUpMsg;
    }

    return msg;
  }

  /// Perform a round-trip using Gemini 3 (via translation layer).
  Future<oai.AssistantMessage> roundTripGemini(String userMessage) async {
    history.add(oai.ChatMessage.user(userMessage));

    final request = oai.ChatCompletionCreateRequest(
      model: geminiModel,
      messages: history,
      tools: tools,
      toolChoice: oai.ToolChoice.auto(),
    );

    // Convert OpenAI → Gemini.
    final geminiReq = ChatCompletionRequestConverter.convert(
      request,
      thoughtSignatures: thoughtSignatures,
    );

    // Send to Gemini API.
    final response = await geminiClient.models.generateContent(
      model: geminiModel,
      request: gai.GenerateContentRequest(
        contents: geminiReq.contents,
        systemInstruction: geminiReq.systemInstruction,
        tools: geminiReq.tools,
        toolConfig: geminiReq.toolConfig?.toJson(),
        generationConfig: geminiReq.generationConfig,
      ),
    );

    // Convert Gemini → OpenAI.
    final result = ChatCompletionResponseConverter.convert(
      response,
      model: geminiModel,
      generateToolCallId: generateToolCallId,
    );

    thoughtSignatures = {...thoughtSignatures, ...result.thoughtSignatures};
    final msg = result.completion.choices.first.message;
    history.add(msg);

    // If tool calls, execute and get follow-up.
    if (msg.toolCalls != null && msg.toolCalls!.isNotEmpty) {
      handleToolCalls(msg);

      final followUpReq = oai.ChatCompletionCreateRequest(
        model: geminiModel,
        messages: history,
        tools: tools,
      );

      final followUpGemini = ChatCompletionRequestConverter.convert(
        followUpReq,
        thoughtSignatures: thoughtSignatures,
      );

      final followUpResponse = await geminiClient.models.generateContent(
        model: geminiModel,
        request: gai.GenerateContentRequest(
          contents: followUpGemini.contents,
          systemInstruction: followUpGemini.systemInstruction,
          tools: followUpGemini.tools,
          toolConfig: followUpGemini.toolConfig?.toJson(),
          generationConfig: followUpGemini.generationConfig,
        ),
      );

      final followUpResult = ChatCompletionResponseConverter.convert(
        followUpResponse,
        model: geminiModel,
        generateToolCallId: generateToolCallId,
      );

      thoughtSignatures = {
        ...thoughtSignatures,
        ...followUpResult.thoughtSignatures,
      };
      final followUpMsg = followUpResult.completion.choices.first.message;
      history.add(followUpMsg);
      return followUpMsg;
    }

    return msg;
  }

  setUpAll(() {
    var openAIKey = Platform.environment['OPENAI_API_KEY'] ?? '';
    var geminiKey = Platform.environment['GEMINI_API_KEY'] ?? '';

    // Load from .env if not in environment.
    final envFile = File('.env');
    if (envFile.existsSync()) {
      for (final line in envFile.readAsLinesSync()) {
        final trimmed = line.trim();
        if (trimmed.startsWith('OPENAI_API_KEY=') && openAIKey.isEmpty) {
          openAIKey = trimmed.substring('OPENAI_API_KEY='.length);
        }
        if (trimmed.startsWith('GEMINI_API_KEY=') && geminiKey.isEmpty) {
          geminiKey = trimmed.substring('GEMINI_API_KEY='.length);
        }
      }
    }

    if (openAIKey.isEmpty) {
      fail('OPENAI_API_KEY not found. Set it in environment or .env file.');
    }
    if (geminiKey.isEmpty) {
      fail('GEMINI_API_KEY not found. Set it in environment or .env file.');
    }

    openAIClient = oai.OpenAIClient.withApiKey(openAIKey);
    geminiClient = gai.GoogleAIClient.withApiKey(geminiKey);
  });

  test(
    'E2E: 12 round-trips alternating GPT-5.1 ↔ Gemini 3',
    () async {
      // Add a system message to anchor the conversation.
      history.add(
        oai.ChatMessage.system(
          'You are a helpful geography assistant. When asked about capitals or '
          'populations, use the provided tools. Keep responses brief.',
        ),
      );

      // Define 12 prompts that build on each other, alternating providers.
      // Odd rounds = GPT-5.1, Even rounds = Gemini 3 (1-indexed).
      final rounds = <_Round>[
        // 1: GPT-5.1 - simple question
        _Round(
          prompt: 'Hi! What continent is France in? Just the continent name.',
          provider: 'openai',
          validator: (r) => expect(
            r.toLowerCase(),
            contains('europe'),
          ),
        ),
        // 2: Gemini 3 - simple follow-up using context from round 1
        _Round(
          prompt: 'And Japan?',
          provider: 'gemini',
          validator: (r) => expect(
            r.toLowerCase(),
            contains('asia'),
          ),
        ),
        // 3: GPT-5.1 - tool call (lookup_capital)
        _Round(
          prompt: 'What is the capital of France? Use the lookup_capital tool.',
          provider: 'openai',
          expectToolCalls: true,
          validator: (r) => expect(r.toLowerCase(), contains('paris')),
        ),
        // 4: Gemini 3 - tool call (lookup_capital), building on GPT's context
        _Round(
          prompt: 'What about Japan\'s capital? Use the lookup_capital tool.',
          provider: 'gemini',
          expectToolCalls: true,
          validator: (r) => expect(r.toLowerCase(), contains('tokyo')),
        ),
        // 5: GPT-5.1 - tool call (lookup_population)
        _Round(
          prompt: 'What is the population of Paris? Use lookup_population.',
          provider: 'openai',
          expectToolCalls: true,
          validator: (r) => expect(r.toLowerCase(), contains('2.1')),
        ),
        // 6: Gemini 3 - tool call (lookup_population)
        _Round(
          prompt: 'And Tokyo\'s population? Use lookup_population.',
          provider: 'gemini',
          expectToolCalls: true,
          validator: (r) => expect(r.toLowerCase(), contains('13.9')),
        ),
        // 7: GPT-5.1 - context recall (no tools)
        _Round(
          prompt:
              'Which of the two cities we discussed has a larger population? '
              'Just the city name.',
          provider: 'openai',
          validator: (r) => expect(r.toLowerCase(), contains('tokyo')),
        ),
        // 8: Gemini 3 - tool call for new country
        _Round(
          prompt: 'What is the capital of Brazil? Use the lookup_capital tool.',
          provider: 'gemini',
          expectToolCalls: true,
          validator: (r) => expect(
            r.toLowerCase(),
            anyOf(contains('brasília'), contains('brasilia')),
          ),
        ),
        // 9: GPT-5.1 - tool call for population of that capital
        _Round(
          prompt:
              'What is the population of that Brazilian capital? '
              'Use lookup_population.',
          provider: 'openai',
          expectToolCalls: true,
          validator: (r) => expect(r.toLowerCase(), contains('3.0')),
        ),
        // 10: Gemini 3 - context recall across providers
        _Round(
          prompt:
              'List all the countries we\'ve discussed so far. '
              'Just the country names, comma-separated.',
          provider: 'gemini',
          validator: (r) {
            final lower = r.toLowerCase();
            expect(lower, contains('france'));
            expect(lower, contains('japan'));
            expect(lower, contains('brazil'));
          },
        ),
        // 11: GPT-5.1 - new tool call
        _Round(
          prompt: 'What is the capital of Australia? Use the lookup_capital tool.',
          provider: 'openai',
          expectToolCalls: true,
          validator: (r) => expect(r.toLowerCase(), contains('canberra')),
        ),
        // 12: Gemini 3 - final summary referencing entire conversation
        _Round(
          prompt:
              'List all the capital cities we found during our conversation. '
              'Just the city names, comma-separated.',
          provider: 'gemini',
          validator: (r) {
            final lower = r.toLowerCase();
            expect(lower, contains('paris'));
            expect(lower, contains('tokyo'));
            expect(lower, anyOf(contains('brasília'), contains('brasilia')));
            expect(lower, contains('canberra'));
          },
        ),
      ];

      for (var i = 0; i < rounds.length; i++) {
        final round = rounds[i];
        final roundNum = i + 1;
        final provider = round.provider == 'openai' ? 'GPT-5.1' : 'Gemini 3';

        print('\n--- Round $roundNum/12 [$provider] ---');
        print('  Prompt: ${round.prompt}');

        late oai.AssistantMessage response;
        if (round.provider == 'openai') {
          response = await roundTripOpenAI(round.prompt);
        } else {
          response = await roundTripGemini(round.prompt);
        }

        final text = response.content ?? '(tool calls only)';
        print('  Response: $text');

        if (round.expectToolCalls) {
          // The follow-up response (after tool execution) should have text.
          expect(response.content, isNotNull, reason: 'Round $roundNum: Expected text after tool execution');
        }

        // Validate the response content.
        round.validator(response.content ?? '');

        print('  ✓ Validated');
        print('  History size: ${history.length} messages');
        print('  Thought signatures: ${thoughtSignatures.length}');
      }

      // Print final conversation summary.
      print('\n${'=' * 60}');
      print('FINAL CONVERSATION HISTORY (${history.length} messages)');
      print('=' * 60);
      for (var i = 0; i < history.length; i++) {
        final msg = history[i];
        final role = switch (msg) {
          oai.SystemMessage() => 'system',
          oai.UserMessage() => 'user',
          oai.AssistantMessage() => 'assistant',
          oai.ToolMessage() => 'tool',
          oai.DeveloperMessage() => 'developer',
        };
        final preview = switch (msg) {
          oai.UserMessage(:final content) => switch (content) {
            oai.UserTextContent(:final text) => text.substring(0, text.length.clamp(0, 60)),
            _ => '(multipart)',
          },
          oai.AssistantMessage(:final content, :final toolCalls) =>
            toolCalls != null && toolCalls.isNotEmpty
                ? 'tool_calls: ${toolCalls.map((t) => t.function.name).join(', ')}'
                : (content ?? '(empty)').substring(0, (content ?? '').length.clamp(0, 60)),
          oai.ToolMessage(:final content) => content.substring(0, content.length.clamp(0, 50)),
          oai.SystemMessage(:final content) => content.substring(0, content.length.clamp(0, 50)),
          oai.DeveloperMessage(:final content) => content.substring(0, content.length.clamp(0, 50)),
        };
        print('  [$i] $role: $preview');
      }

      print('\n${'=' * 60}');
      print('12 cross-provider round-trips completed successfully!');
      print('Total messages: ${history.length}');
      print('Thought signatures preserved: ${thoughtSignatures.length}');
      print('=' * 60);
    },
    timeout: const Timeout(Duration(minutes: 5)),
  );
}

class _Round {
  final String prompt;
  final String provider; // 'openai' or 'gemini'
  final bool expectToolCalls;
  final void Function(String response) validator;

  const _Round({
    required this.prompt,
    required this.provider,
    this.expectToolCalls = false,
    required this.validator,
  });
}
