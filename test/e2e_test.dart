import 'dart:convert';
import 'dart:io';

import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:test/test.dart';

/// Live end-to-end test that performs 4 round-trips with Gemini 3,
/// verifying thought signature preservation across tool-calling turns.
///
/// Requires GEMINI_API_KEY environment variable or .env file.
void main() {
  late gai.GoogleAIClient client;
  late String apiKey;

  // The model to test against.
  const model = 'gemini-3-flash-preview';

  // Track the full OpenAI-format conversation history.
  final openAIHistory = <oai.ChatMessage>[];

  // Track thought signatures across turns.
  var thoughtSignatures = <String, String>{};

  // Tool call ID counter.
  var toolCallCounter = 0;
  String generateToolCallId() => 'call_${toolCallCounter++}';

  // Tool definitions in OpenAI format.
  final tools = [
    oai.Tool.function(
      name: 'get_weather',
      description: 'Get the current weather for a location.',
      parameters: {
        'type': 'object',
        'properties': {
          'location': {
            'type': 'string',
            'description': 'The city and state, e.g. "San Francisco, CA"',
          },
          'unit': {
            'type': 'string',
            'description': 'Temperature unit',
            'enum': ['celsius', 'fahrenheit'],
          },
        },
        'required': ['location'],
      },
    ),
    oai.Tool.function(
      name: 'get_time',
      description: 'Get the current time in a timezone.',
      parameters: {
        'type': 'object',
        'properties': {
          'timezone': {
            'type': 'string',
            'description':
                'The timezone, e.g. "America/New_York", "Europe/London"',
          },
        },
        'required': ['timezone'],
      },
    ),
  ];

  /// Simulates executing a tool call locally.
  String executeToolCall(String name, Map<String, dynamic> args) {
    return switch (name) {
      'get_weather' => jsonEncode({
          'temperature': 72,
          'unit': args['unit'] ?? 'fahrenheit',
          'condition': 'sunny',
          'location': args['location'],
        }),
      'get_time' => jsonEncode({
          'time': '2026-03-09T14:30:00',
          'timezone': args['timezone'],
        }),
      _ => jsonEncode({'error': 'Unknown function: $name'}),
    };
  }

  /// Performs a single round-trip:
  /// 1. Converts OpenAI history → Gemini format
  /// 2. Sends to Gemini API
  /// 3. Converts response → OpenAI format
  /// 4. If tool calls, executes them and adds results to history
  /// Returns the final assistant message text (or null if only tool calls).
  Future<String?> performRoundTrip({
    required String userMessage,
    bool expectToolCalls = false,
  }) async {
    // Add user message to OpenAI history.
    openAIHistory.add(oai.ChatMessage.user(userMessage));

    // Build the request in OpenAI format.
    final request = oai.ChatCompletionCreateRequest(
      model: model,
      messages: openAIHistory,
      tools: tools,
      toolChoice: oai.ToolChoice.auto(),
    );

    // Convert OpenAI request → Gemini request.
    final geminiRequest = ChatCompletionRequestConverter.convert(
      request,
      thoughtSignatures: thoughtSignatures,
    );

    // Send to Gemini API.
    final response = await client.models.generateContent(
      model: model,
      request: gai.GenerateContentRequest(
        contents: geminiRequest.contents,
        systemInstruction: geminiRequest.systemInstruction,
        tools: geminiRequest.tools,
        toolConfig: geminiRequest.toolConfig != null
            ? geminiRequest.toolConfig!.toJson()
            : null,
        generationConfig: geminiRequest.generationConfig,
      ),
    );

    // Convert Gemini response → OpenAI format.
    final result = ChatCompletionResponseConverter.convert(
      response,
      model: model,
      generateToolCallId: generateToolCallId,
    );

    // Merge thought signatures.
    thoughtSignatures = {
      ...thoughtSignatures,
      ...result.thoughtSignatures,
    };

    final assistantMsg = result.completion.choices.first.message;
    openAIHistory.add(assistantMsg);

    // If the model made tool calls, execute them.
    if (assistantMsg.toolCalls != null && assistantMsg.toolCalls!.isNotEmpty) {
      expect(expectToolCalls, isTrue, reason: 'Unexpected tool calls');

      for (final toolCall in assistantMsg.toolCalls!) {
        final args =
            jsonDecode(toolCall.function.arguments) as Map<String, dynamic>;
        final toolResult = executeToolCall(toolCall.function.name, args);

        // Add tool result to OpenAI history.
        openAIHistory.add(
          oai.ChatMessage.tool(
            toolCallId: toolCall.id,
            content: toolResult,
          ),
        );
      }

      // Send follow-up with tool results.
      final followUpRequest = oai.ChatCompletionCreateRequest(
        model: model,
        messages: openAIHistory,
        tools: tools,
      );

      final followUpGemini = ChatCompletionRequestConverter.convert(
        followUpRequest,
        thoughtSignatures: thoughtSignatures,
      );

      final followUpResponse = await client.models.generateContent(
        model: model,
        request: gai.GenerateContentRequest(
          contents: followUpGemini.contents,
          systemInstruction: followUpGemini.systemInstruction,
          tools: followUpGemini.tools,
          toolConfig: followUpGemini.toolConfig != null
              ? followUpGemini.toolConfig!.toJson()
              : null,
          generationConfig: followUpGemini.generationConfig,
        ),
      );

      final followUpResult = ChatCompletionResponseConverter.convert(
        followUpResponse,
        model: model,
        generateToolCallId: generateToolCallId,
      );

      thoughtSignatures = {
        ...thoughtSignatures,
        ...followUpResult.thoughtSignatures,
      };

      final followUpMsg = followUpResult.completion.choices.first.message;
      openAIHistory.add(followUpMsg);

      return followUpMsg.content;
    }

    return assistantMsg.content;
  }

  setUpAll(() {
    // Load API key from environment or .env file.
    apiKey = Platform.environment['GEMINI_API_KEY'] ?? '';
    if (apiKey.isEmpty) {
      final envFile = File('.env');
      if (envFile.existsSync()) {
        for (final line in envFile.readAsLinesSync()) {
          final trimmed = line.trim();
          if (trimmed.startsWith('GEMINI_API_KEY=')) {
            apiKey = trimmed.substring('GEMINI_API_KEY='.length);
          }
        }
      }
    }
    if (apiKey.isEmpty) {
      fail(
        'GEMINI_API_KEY not found. Set it as environment variable or in .env file.',
      );
    }

    client = gai.GoogleAIClient.withApiKey(apiKey);
  });

  test(
    'E2E: 4 round-trips with tool calling and thought signature preservation',
    () async {
      // -----------------------------------------------------------------------
      // Round-trip 1: Simple text question (no tools).
      // -----------------------------------------------------------------------
      print('\n--- Round-trip 1: Simple text question ---');
      final response1 = await performRoundTrip(
        userMessage: 'What is 2 + 2? Reply with just the number.',
      );
      print('Response: $response1');
      expect(response1, isNotNull);
      expect(response1, contains('4'));
      expect(openAIHistory.length, 2); // user + assistant
      print('History length: ${openAIHistory.length}');
      print('Thought signatures: ${thoughtSignatures.keys.toList()}');

      // -----------------------------------------------------------------------
      // Round-trip 2: Trigger a tool call (get_weather).
      // -----------------------------------------------------------------------
      print('\n--- Round-trip 2: Tool call (get_weather) ---');
      final response2 = await performRoundTrip(
        userMessage:
            'What is the weather in San Francisco? Use the get_weather tool.',
        expectToolCalls: true,
      );
      print('Response: $response2');
      expect(response2, isNotNull);
      // The response should mention the weather data we returned.
      expect(
        response2!.toLowerCase(),
        anyOf(contains('72'), contains('sunny'), contains('san francisco')),
      );
      print('History length: ${openAIHistory.length}');
      print('Thought signatures: ${thoughtSignatures.keys.toList()}');

      // -----------------------------------------------------------------------
      // Round-trip 3: Another tool call (get_time), testing signature
      // preservation across multiple tool-calling turns.
      // -----------------------------------------------------------------------
      print('\n--- Round-trip 3: Tool call (get_time) ---');
      final response3 = await performRoundTrip(
        userMessage:
            'What time is it in New York? Use the get_time tool.',
        expectToolCalls: true,
      );
      print('Response: $response3');
      expect(response3, isNotNull);
      expect(
        response3!.toLowerCase(),
        anyOf(contains('14:30'), contains('2:30'), contains('new york')),
      );
      print('History length: ${openAIHistory.length}');
      print('Thought signatures: ${thoughtSignatures.keys.toList()}');

      // -----------------------------------------------------------------------
      // Round-trip 4: Follow-up referencing previous context (no tools).
      // This validates that the full conversation history round-trips
      // correctly through the OpenAI format.
      // -----------------------------------------------------------------------
      print('\n--- Round-trip 4: Context recall (no tools) ---');
      final response4 = await performRoundTrip(
        userMessage:
            'Based on our conversation, which city did I ask about the '
            'weather for? Reply with just the city name.',
      );
      print('Response: $response4');
      expect(response4, isNotNull);
      expect(response4!.toLowerCase(), contains('san francisco'));
      print('History length: ${openAIHistory.length}');

      // -----------------------------------------------------------------------
      // Verify the full conversation stored in OpenAI format.
      // -----------------------------------------------------------------------
      print('\n--- Final conversation history ---');
      for (var i = 0; i < openAIHistory.length; i++) {
        final msg = openAIHistory[i];
        final role = switch (msg) {
          oai.UserMessage() => 'user',
          oai.AssistantMessage() => 'assistant',
          oai.ToolMessage() => 'tool',
          oai.SystemMessage() => 'system',
          oai.DeveloperMessage() => 'developer',
        };
        final preview = switch (msg) {
          oai.UserMessage(:final content) => switch (content) {
              oai.UserTextContent(:final text) => text,
              _ => '(multipart)',
            },
          oai.AssistantMessage(:final content, :final toolCalls) =>
            toolCalls != null
                ? 'tool_calls: ${toolCalls.map((t) => t.function.name).join(', ')}'
                : content ?? '(empty)',
          oai.ToolMessage(:final content) =>
            content.substring(0, content.length.clamp(0, 50)),
          oai.SystemMessage(:final content) => content,
          oai.DeveloperMessage(:final content) => content,
        };
        print('  [$i] $role: $preview');
      }

      print('\nAll 4 round-trips completed successfully!');
      print('Total messages in OpenAI history: ${openAIHistory.length}');
      print(
        'Thought signatures preserved: ${thoughtSignatures.length}',
      );
    },
    timeout: const Timeout(Duration(minutes: 3)),
  );
}
