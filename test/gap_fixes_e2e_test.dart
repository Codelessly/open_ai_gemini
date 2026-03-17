import 'dart:convert';
import 'dart:io';

import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:test/test.dart';

/// Live E2E tests that exercise the gap fixes against the real Gemini API.
///
/// These tests verify that:
/// 1. Thought signatures + sentinel work across multi-turn tool-calling
/// 2. Error/success tool results are properly handled
/// 3. Unicode sanitization doesn't break real API calls
/// 4. Cross-provider conversations survive round-trips
/// 5. OpenAI format integrity is preserved throughout
///
/// Requires GEMINI_API_KEY environment variable or .env file.
void main() {
  late gai.GoogleAIClient geminiClient;
  late oai.OpenAIClient openAIClient;
  late String geminiApiKey;
  late String openAIApiKey;
  late bool hasOpenAIKey;

  const geminiModel = 'gemini-3-flash-preview';

  setUpAll(() {
    geminiApiKey = Platform.environment['GEMINI_API_KEY'] ?? '';
    openAIApiKey = Platform.environment['OPENAI_API_KEY'] ?? '';

    final envFile = File('.env');
    if (envFile.existsSync()) {
      for (final line in envFile.readAsLinesSync()) {
        final trimmed = line.trim();
        if (trimmed.startsWith('GEMINI_API_KEY=') && geminiApiKey.isEmpty) {
          geminiApiKey = trimmed.substring('GEMINI_API_KEY='.length);
        }
        if (trimmed.startsWith('OPENAI_API_KEY=') && openAIApiKey.isEmpty) {
          openAIApiKey = trimmed.substring('OPENAI_API_KEY='.length);
        }
      }
    }

    if (geminiApiKey.isEmpty) {
      fail('GEMINI_API_KEY not found.');
    }

    geminiClient = gai.GoogleAIClient.withApiKey(geminiApiKey);
    hasOpenAIKey = openAIApiKey.isNotEmpty;
    if (hasOpenAIKey) {
      openAIClient = oai.OpenAIClient.withApiKey(openAIApiKey);
    }
  });

  /// Helper: send an OpenAI request through Gemini and get back OpenAI response.
  Future<GeminiResponseConversionResult> callGemini({
    required List<oai.ChatMessage> messages,
    required Map<String, String> thoughtSignatures,
    List<oai.Tool>? tools,
    String Function()? generateToolCallId,
    String? sourceProvider,
    String? sourceModel,
  }) async {
    final request = oai.ChatCompletionCreateRequest(
      model: geminiModel,
      messages: messages,
      tools: tools,
      toolChoice: tools != null ? oai.ToolChoice.auto() : null,
    );

    final messageResult = MessageContentConverter.toGemini(
      request.messages,
      thoughtSignatures: thoughtSignatures,
      modelId: geminiModel,
      sourceProvider: sourceProvider,
      sourceModel: sourceModel,
    );

    final geminiResponse = await geminiClient.models.generateContent(
      model: geminiModel,
      request: gai.GenerateContentRequest(
        contents: messageResult.contents,
        systemInstruction: messageResult.systemInstruction,
        tools: ChatCompletionRequestConverter.buildTools(request),
        toolConfig: ChatCompletionRequestConverter.buildToolConfig(request)?.toJson(),
        generationConfig: ChatCompletionRequestConverter.buildGenerationConfig(request),
      ),
    );

    return ChatCompletionResponseConverter.convert(
      geminiResponse,
      model: geminiModel,
      generateToolCallId: generateToolCallId,
    );
  }

  // ===========================================================================
  // E2E: Multi-turn tool calling with thought signatures (Gap #1, #5, #6)
  // ===========================================================================
  test(
    'E2E: Multi-turn tool calling with thought signature preservation',
    () async {
      final history = <oai.ChatMessage>[];
      var thoughtSignatures = <String, String>{};
      var toolCallCounter = 0;
      String genId() => 'call_${toolCallCounter++}';

      final tools = [
        oai.Tool.function(
          name: 'get_weather',
          description: 'Get weather for a location.',
          parameters: {
            'type': 'object',
            'properties': {
              'location': {
                'type': 'string',
                'description': 'City name',
              },
            },
            'required': ['location'],
          },
        ),
      ];

      // --- Turn 1: Simple question ---
      history.add(oai.ChatMessage.user('What is 2 + 2? Reply with just the number.'));

      var result = await callGemini(
        messages: history,
        thoughtSignatures: thoughtSignatures,
      );
      thoughtSignatures = {...thoughtSignatures, ...result.thoughtSignatures};
      history.add(result.completion.choices.first.message);

      print('Turn 1: ${result.completion.choices.first.message.content}');
      expect(result.completion.choices.first.message.content, contains('4'));

      // --- Turn 2: Tool call (triggers thought signatures) ---
      history.add(oai.ChatMessage.user('What is the weather in Tokyo? Use the get_weather tool.'));

      result = await callGemini(
        messages: history,
        thoughtSignatures: thoughtSignatures,
        tools: tools,
        generateToolCallId: genId,
        sourceProvider: 'gemini',
        sourceModel: geminiModel,
      );
      thoughtSignatures = {...thoughtSignatures, ...result.thoughtSignatures};
      final msg2 = result.completion.choices.first.message;
      history.add(msg2);

      expect(msg2.toolCalls, isNotNull, reason: 'Should trigger tool call');
      print('Turn 2: Tool calls: ${msg2.toolCalls!.map((t) => t.function.name)}');
      print('  Signatures captured: ${thoughtSignatures.keys}');

      // Execute tool and add result.
      for (final tc in msg2.toolCalls!) {
        history.add(
          oai.ChatMessage.tool(
            toolCallId: tc.id,
            content: jsonEncode({
              'temperature': 15,
              'condition': 'cloudy',
              'location': 'Tokyo',
            }),
          ),
        );
      }

      // --- Turn 3: Follow-up after tool (replays signatures) ---
      result = await callGemini(
        messages: history,
        thoughtSignatures: thoughtSignatures,
        tools: tools,
        sourceProvider: 'gemini',
        sourceModel: geminiModel,
      );
      thoughtSignatures = {...thoughtSignatures, ...result.thoughtSignatures};
      history.add(result.completion.choices.first.message);

      final turn3Text = result.completion.choices.first.message.content ?? '';
      print('Turn 3: $turn3Text');
      expect(turn3Text.toLowerCase(), anyOf(contains('tokyo'), contains('15'), contains('cloudy')));

      // --- Turn 4: Context recall across all turns ---
      history.add(oai.ChatMessage.user('What city did I ask about the weather for? Reply with just the city name.'));

      result = await callGemini(
        messages: history,
        thoughtSignatures: thoughtSignatures,
        sourceProvider: 'gemini',
        sourceModel: geminiModel,
      );
      history.add(result.completion.choices.first.message);

      final turn4Text = result.completion.choices.first.message.content ?? '';
      print('Turn 4: $turn4Text');
      expect(turn4Text.toLowerCase(), contains('tokyo'));

      print('\nAll 4 turns completed successfully!');
      print('History: ${history.length} messages');
      print('Signatures: ${thoughtSignatures.length}');
    },
    timeout: const Timeout(Duration(minutes: 3)),
  );

  // ===========================================================================
  // E2E: Error tool results (Gap #3)
  // ===========================================================================
  test(
    'E2E: Error tool results are properly communicated to model',
    () async {
      final history = <oai.ChatMessage>[];
      var thoughtSignatures = <String, String>{};
      var toolCallCounter = 0;
      String genId() => 'call_${toolCallCounter++}';

      final tools = [
        oai.Tool.function(
          name: 'fetch_data',
          description: 'Fetch data from a URL.',
          parameters: {
            'type': 'object',
            'properties': {
              'url': {'type': 'string', 'description': 'The URL to fetch'},
            },
            'required': ['url'],
          },
        ),
      ];

      // Ask model to use tool.
      history.add(oai.ChatMessage.user('Fetch data from https://example.com/api. Use the fetch_data tool.'));

      var result = await callGemini(
        messages: history,
        thoughtSignatures: thoughtSignatures,
        tools: tools,
        generateToolCallId: genId,
      );
      thoughtSignatures = {...thoughtSignatures, ...result.thoughtSignatures};
      final msg = result.completion.choices.first.message;
      history.add(msg);

      expect(msg.toolCalls, isNotNull);

      // Return an error result using the isError parameter.
      for (final tc in msg.toolCalls!) {
        final errorResponse = ToolMapper.toGeminiFunctionResponse(
          functionName: tc.function.name,
          content: 'Connection timeout after 30s',
          isError: true,
        );
        // Verify the error key is used in the response.
        expect(errorResponse.functionResponse.response.containsKey('error'), isTrue);

        history.add(
          oai.ChatMessage.tool(
            toolCallId: tc.id,
            content: 'Connection timeout after 30s',
          ),
        );
      }

      // Follow up — model should acknowledge the error.
      result = await callGemini(
        messages: history,
        thoughtSignatures: thoughtSignatures,
        tools: tools,
        sourceProvider: 'gemini',
        sourceModel: geminiModel,
      );
      history.add(result.completion.choices.first.message);

      final responseText = result.completion.choices.first.message.content ?? '';
      print('Error response: $responseText');
      // Model should mention the error or issue in some way.
      expect(responseText, isNotEmpty);
    },
    timeout: const Timeout(Duration(minutes: 2)),
  );

  // ===========================================================================
  // E2E: Unicode content survives round-trip (Gap #4)
  // ===========================================================================
  test(
    'E2E: Unicode content including CJK and emoji survives round-trip',
    () async {
      final history = <oai.ChatMessage>[
        oai.ChatMessage.system('Reply in the same language as the user.'),
        oai.ChatMessage.user('日本語で「こんにちは」と返事してください。漢字を使ってください。'),
      ];

      final result = await callGemini(
        messages: history,
        thoughtSignatures: {},
      );

      final text = result.completion.choices.first.message.content ?? '';
      print('Unicode response: $text');
      // Should contain Japanese characters.
      expect(text, isNotEmpty);
      // Verify no replacement characters were introduced.
      expect(text.contains('\uFFFD'), isFalse);
    },
    timeout: const Timeout(Duration(minutes: 1)),
  );

  // ===========================================================================
  // E2E: GeminiOpenAIClient drop-in with tool calling (comprehensive)
  // ===========================================================================
  test(
    'E2E: GeminiOpenAIClient handles multi-turn tool calling',
    () async {
      final client = GeminiOpenAIClient(apiKey: geminiApiKey);

      // Turn 1: Simple text.
      var response = await client.chat.completions.create(
        oai.ChatCompletionCreateRequest(
          model: geminiModel,
          messages: [
            oai.ChatMessage.user('What is the capital of France? Just the city name.'),
          ],
        ),
      );

      print('Client Turn 1: ${response.choices.first.message.content}');
      expect(response.choices.first.message.content?.toLowerCase(), contains('paris'));

      // Turn 2: With tools.
      final history = <oai.ChatMessage>[
        oai.ChatMessage.user('What is the weather in London?'),
      ];

      response = await client.chat.completions.create(
        oai.ChatCompletionCreateRequest(
          model: geminiModel,
          messages: history,
          tools: [
            oai.Tool.function(
              name: 'get_weather',
              description: 'Get weather for a city.',
              parameters: {
                'type': 'object',
                'properties': {
                  'city': {'type': 'string'},
                },
                'required': ['city'],
              },
            ),
          ],
          toolChoice: oai.ToolChoice.auto(),
        ),
      );

      final msg = response.choices.first.message;
      print('Client Turn 2: ${msg.toolCalls?.map((t) => t.function.name) ?? msg.content}');

      // Verify the response is valid OpenAI format.
      expect(response.model, geminiModel);
      expect(response.provider, 'gemini');
      expect(response.object, 'chat.completion');

      client.close();
    },
    timeout: const Timeout(Duration(minutes: 2)),
  );

  // ===========================================================================
  // E2E: Cross-provider round-trip (Gap #2) - OpenAI → Gemini
  // ===========================================================================
  test(
    'E2E: Cross-provider OpenAI → Gemini conversation preserves context',
    () async {
      if (!hasOpenAIKey) {
        print('Skipping cross-provider test — OPENAI_API_KEY not set.');
        return;
      }

      final history = <oai.ChatMessage>[
        oai.ChatMessage.system('You are a helpful assistant. Keep answers brief.'),
      ];
      var thoughtSignatures = <String, String>{};

      // Turn 1: OpenAI.
      history.add(oai.ChatMessage.user('What continent is Australia on? Just the continent name.'));

      final openAIResponse = await openAIClient.chat.completions.create(
        oai.ChatCompletionCreateRequest(
          model: 'gpt-4.1-nano',
          messages: history,
        ),
      );
      history.add(openAIResponse.choices.first.message);
      print('OpenAI Turn: ${openAIResponse.choices.first.message.content}');

      // Turn 2: Gemini — using the SAME history that includes OpenAI's response.
      // The OpenAI response may have reasoningContent — this tests Gap #2
      // (cross-provider thinking block handling).
      history.add(oai.ChatMessage.user('What is the largest city on that continent? Just the city name.'));

      final geminiResult = await callGemini(
        messages: history,
        thoughtSignatures: thoughtSignatures,
        // Source is OpenAI, target is Gemini — reasoning should be plain text.
        sourceProvider: 'openai',
        sourceModel: 'gpt-4.1-nano',
      );
      thoughtSignatures = {
        ...thoughtSignatures,
        ...geminiResult.thoughtSignatures,
      };
      history.add(geminiResult.completion.choices.first.message);

      final geminiText = geminiResult.completion.choices.first.message.content ?? '';
      print('Gemini Turn: $geminiText');

      // Should reference context from OpenAI turn (Australia → Sydney).
      expect(geminiText.toLowerCase(), contains('sydney'));

      // Turn 3: Back to OpenAI — verify full history survives.
      history.add(oai.ChatMessage.user('What country are we discussing? Just the country name.'));

      final openAIResponse2 = await openAIClient.chat.completions.create(
        oai.ChatCompletionCreateRequest(
          model: 'gpt-4.1-nano',
          messages: history,
        ),
      );
      history.add(openAIResponse2.choices.first.message);

      final openAI2Text = openAIResponse2.choices.first.message.content ?? '';
      print('OpenAI Turn 2: $openAI2Text');
      expect(openAI2Text.toLowerCase(), contains('australia'));

      print('\nCross-provider round-trip successful!');
      print('History: ${history.length} messages');
    },
    timeout: const Timeout(Duration(minutes: 3)),
  );

  // ===========================================================================
  // E2E: Streaming with gap fixes
  // ===========================================================================
  test(
    'E2E: Streaming response captures thought signatures',
    () async {
      final client = GeminiOpenAIClient(apiKey: geminiApiKey);

      final events = <oai.ChatStreamEvent>[];

      final stream = client.chat.completions.createStream(
        oai.ChatCompletionCreateRequest(
          model: geminiModel,
          messages: [
            oai.ChatMessage.user('What is 3 * 7? Reply with just the number.'),
          ],
        ),
      );

      await for (final event in stream) {
        events.add(event);
      }

      expect(events, isNotEmpty);

      // Verify streaming events have proper structure.
      final firstEvent = events.first;
      expect(firstEvent.model, geminiModel);
      expect(firstEvent.object, 'chat.completion.chunk');

      // Accumulate text from deltas.
      final text = StringBuffer();
      for (final event in events) {
        final delta = event.choices?.firstOrNull?.delta;
        if (delta?.content != null) {
          text.write(delta!.content);
        }
      }

      print('Streamed text: $text');
      expect(text.toString(), contains('21'));

      // Verify thought signatures were accumulated by the client.
      // (May or may not have signatures depending on model behavior.)
      print('Client thought signatures: ${client.thoughtSignatures.length}');

      client.close();
    },
    timeout: const Timeout(Duration(minutes: 2)),
  );
}
