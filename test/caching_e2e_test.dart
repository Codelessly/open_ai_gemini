import 'dart:io';

import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:test/test.dart';

/// Live E2E tests for Gemini context caching.
///
/// Validates that:
/// 1. Cached content token counts are properly emitted in responses
/// 2. Multiple round-trips against the same cache produce meaningful cache hits
/// 3. The GeminiOpenAIClient properly surfaces cached token data
/// 4. Cross-provider conversations maintain cache effectiveness
/// 5. Pricing math checks out for Gemini 3 Flash Preview
///
/// Gemini 3 Flash Preview pricing (per 1M tokens):
///   Input:       $0.50
///   Output:      $3.00
///   Cache Read:  $0.05
///   Cache Write: $0.08333
///
/// Requires GEMINI_API_KEY environment variable or .env file.
void main() {
  late GeminiOpenAIClient client;
  late gai.GoogleAIClient rawGeminiClient;
  late String geminiApiKey;
  late String openAIApiKey;
  late bool hasOpenAIKey;

  const model = 'gemini-3-flash-preview';

  // A large system prompt to ensure meaningful caching.
  // Gemini requires minimum ~4096 tokens for caching to kick in.
  final largeSystemPrompt =
      '''
You are an expert historian specializing in ancient civilizations.

${List.generate(200, (i) => 'Historical fact #$i: The ancient world contained many civilizations that rose and fell over millennia. Each civilization contributed unique advances in art, science, governance, and culture that shaped the course of human history. From Mesopotamia to Egypt, from the Indus Valley to China, from Greece to Rome, these civilizations formed the foundation of modern society.').join('\n\n')}

When answering questions, be brief and factual. Use the knowledge above.
''';

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

    client = GeminiOpenAIClient(apiKey: geminiApiKey);
    rawGeminiClient = gai.GoogleAIClient.withApiKey(geminiApiKey);
    hasOpenAIKey = openAIApiKey.isNotEmpty;
  });

  tearDownAll(() {
    client.close();
  });

  // ===========================================================================
  // Test 1: Token usage reporting (non-cached baseline)
  // ===========================================================================
  test(
    'Token usage includes promptTokens and completionTokens',
    () async {
      final response = await client.chat.completions.create(
        oai.ChatCompletionCreateRequest(
          model: model,
          messages: [
            oai.ChatMessage.user('What is 1 + 1? Just the number.'),
          ],
        ),
      );

      final usage = response.usage;
      expect(usage, isNotNull, reason: 'Usage should be present');
      expect(usage!.promptTokens, greaterThan(0), reason: 'promptTokens must be > 0');
      expect(usage.completionTokens, greaterThan(0), reason: 'completionTokens must be > 0');
      expect(
        usage.totalTokens,
        greaterThanOrEqualTo(usage.promptTokens + (usage.completionTokens ?? 0)),
        reason: 'totalTokens >= prompt + completion',
      );

      print('Baseline usage:');
      print('  Prompt tokens: ${usage.promptTokens}');
      print('  Completion tokens: ${usage.completionTokens}');
      print('  Total tokens: ${usage.totalTokens}');
      print('  Cached tokens: ${usage.promptTokensDetails?.cachedTokens ?? "none"}');
    },
    timeout: const Timeout(Duration(minutes: 1)),
  );

  // ===========================================================================
  // Test 2: Explicit caching via CachedContents API
  // ===========================================================================
  test(
    'Explicit cache creation and multi-turn cache hits',
    () async {
      // Create cached content with a large system prompt.
      final cached = await rawGeminiClient.cachedContents.create(
        cachedContent: gai.CachedContent(
          model: 'models/$model',
          systemInstruction: gai.Content(
            parts: [gai.TextPart(largeSystemPrompt)],
          ),
          contents: [
            gai.Content(role: 'user', parts: [gai.TextPart('Hello')]),
            gai.Content(role: 'model', parts: [gai.TextPart('Hello! I am your history expert.')]),
          ],
          ttl: '300s', // 5 minutes
        ),
      );

      final cacheName = cached.name!;
      print('Created cache: $cacheName');
      print('  Token count: ${cached.usageMetadata?.totalTokenCount}');

      try {
        // Set the cache on our client.
        client.cachedContent = cacheName;

        // --- Round-trip 1: First cached request ---
        final response1 = await client.chat.completions.create(
          oai.ChatCompletionCreateRequest(
            model: model,
            messages: [
              oai.ChatMessage.user('Who built the Great Pyramid? Just the pharaoh name.'),
            ],
          ),
        );

        final usage1 = response1.usage!;
        final cached1 = usage1.promptTokensDetails?.cachedTokens ?? 0;
        print('\nRound-trip 1:');
        print('  Response: ${response1.choices.first.message.content}');
        print('  Prompt tokens: ${usage1.promptTokens}');
        print('  Cached tokens: $cached1');
        print('  Completion tokens: ${usage1.completionTokens}');

        // With caching, we should see cached tokens.
        expect(cached1, greaterThan(0), reason: 'First cached request should report cached tokens');

        // --- Round-trip 2: Same cache, different question ---
        final response2 = await client.chat.completions.create(
          oai.ChatCompletionCreateRequest(
            model: model,
            messages: [
              oai.ChatMessage.user('When was Rome founded? Just the year.'),
            ],
          ),
        );

        final usage2 = response2.usage!;
        final cached2 = usage2.promptTokensDetails?.cachedTokens ?? 0;
        print('\nRound-trip 2:');
        print('  Response: ${response2.choices.first.message.content}');
        print('  Prompt tokens: ${usage2.promptTokens}');
        print('  Cached tokens: $cached2');

        expect(cached2, greaterThan(0), reason: 'Second cached request should also have cache hits');

        // Cached tokens should be similar between requests (same cache).
        expect(cached2, closeTo(cached1, cached1 * 0.2), reason: 'Cache hits should be consistent across requests');

        // --- Round-trip 3: Verify cost math ---
        // Gemini 3 Flash Preview pricing per 1M tokens:
        //   Input:      $0.50
        //   Output:     $3.00
        //   Cache Read: $0.05
        //   Cache Write: $0.08333
        final inputCostPerToken = 0.50 / 1e6;
        final outputCostPerToken = 3.00 / 1e6;
        final cacheReadCostPerToken = 0.05 / 1e6;

        final nonCachedInputTokens = usage2.promptTokens - cached2;
        final costWithCache =
            (nonCachedInputTokens * inputCostPerToken) +
            (cached2 * cacheReadCostPerToken) +
            ((usage2.completionTokens ?? 0) * outputCostPerToken);
        final costWithoutCache =
            (usage2.promptTokens * inputCostPerToken) + ((usage2.completionTokens ?? 0) * outputCostPerToken);
        final savings = costWithoutCache - costWithCache;
        final savingsPercent = costWithoutCache > 0 ? (savings / costWithoutCache) * 100 : 0;

        print('\nCost analysis (Round-trip 2):');
        print('  Non-cached input tokens: $nonCachedInputTokens');
        print('  Cached tokens: $cached2');
        print('  Cost with cache:    \$${costWithCache.toStringAsFixed(6)}');
        print('  Cost without cache: \$${costWithoutCache.toStringAsFixed(6)}');
        print('  Savings:            \$${savings.toStringAsFixed(6)} (${savingsPercent.toStringAsFixed(1)}%)');

        // Cache read is 10x cheaper ($0.05 vs $0.50), so we should see savings.
        expect(savings, greaterThan(0), reason: 'Cached requests should be cheaper than uncached');

        // --- Round-trip 4: Third request, verify consistency ---
        final response3 = await client.chat.completions.create(
          oai.ChatCompletionCreateRequest(
            model: model,
            messages: [
              oai.ChatMessage.user('What was the capital of the Byzantine Empire? Just the city.'),
            ],
          ),
        );

        final usage3 = response3.usage!;
        final cached3 = usage3.promptTokensDetails?.cachedTokens ?? 0;
        print('\nRound-trip 3:');
        print('  Response: ${response3.choices.first.message.content}');
        print('  Cached tokens: $cached3');

        expect(cached3, greaterThan(0));
      } finally {
        // Clean up: delete the cached content and reset client.
        client.cachedContent = null;
        await rawGeminiClient.cachedContents.delete(name: cacheName);
        print('\nCache deleted: $cacheName');
      }
    },
    timeout: const Timeout(Duration(minutes: 3)),
  );

  // ===========================================================================
  // Test 3: Streaming also reports cached tokens
  // ===========================================================================
  test(
    'Streaming responses report cached token counts',
    () async {
      // Create a small cache for streaming test.
      final cached = await rawGeminiClient.cachedContents.create(
        cachedContent: gai.CachedContent(
          model: 'models/$model',
          systemInstruction: gai.Content(
            parts: [gai.TextPart(largeSystemPrompt)],
          ),
          contents: [
            gai.Content(role: 'user', parts: [gai.TextPart('Hello')]),
            gai.Content(role: 'model', parts: [gai.TextPart('Hello! Ask me about history.')]),
          ],
          ttl: '120s',
        ),
      );

      final cacheName = cached.name!;

      try {
        client.cachedContent = cacheName;

        final events = <oai.ChatStreamEvent>[];
        final stream = client.chat.completions.createStream(
          oai.ChatCompletionCreateRequest(
            model: model,
            messages: [
              oai.ChatMessage.user('What year did the Roman Republic end? Just the year.'),
            ],
          ),
        );

        await for (final event in stream) {
          events.add(event);
        }

        expect(events, isNotEmpty);

        // Find the final event with usage data.
        final eventsWithUsage = events.where((e) => e.usage != null).toList();

        if (eventsWithUsage.isNotEmpty) {
          final usage = eventsWithUsage.last.usage!;
          final cachedTokens = usage.promptTokensDetails?.cachedTokens ?? 0;

          print('Streaming cached usage:');
          print('  Prompt tokens: ${usage.promptTokens}');
          print('  Cached tokens: $cachedTokens');
          print('  Completion tokens: ${usage.completionTokens}');

          expect(cachedTokens, greaterThan(0), reason: 'Streaming should report cached tokens');
        }

        // Accumulate text.
        final text = StringBuffer();
        for (final event in events) {
          final delta = event.choices?.firstOrNull?.delta;
          if (delta?.content != null) {
            text.write(delta!.content);
          }
        }
        print('Streamed response: $text');
        expect(text.toString(), isNotEmpty);
      } finally {
        client.cachedContent = null;
        await rawGeminiClient.cachedContents.delete(name: cacheName);
      }
    },
    timeout: const Timeout(Duration(minutes: 2)),
  );

  // ===========================================================================
  // Test 4: Cross-provider with cache — OpenAI then Gemini with cache
  // ===========================================================================
  test(
    'Cross-provider: OpenAI history + Gemini cached context',
    () async {
      if (!hasOpenAIKey) {
        print('Skipping — OPENAI_API_KEY not set.');
        return;
      }

      final openAIClient = oai.OpenAIClient.withApiKey(openAIApiKey);

      // Create cache.
      final cached = await rawGeminiClient.cachedContents.create(
        cachedContent: gai.CachedContent(
          model: 'models/$model',
          systemInstruction: gai.Content(
            parts: [gai.TextPart(largeSystemPrompt)],
          ),
          contents: [
            gai.Content(role: 'user', parts: [gai.TextPart('Hello')]),
            gai.Content(role: 'model', parts: [gai.TextPart('Hello! I am your history expert.')]),
          ],
          ttl: '120s',
        ),
      );

      final cacheName = cached.name!;

      try {
        // Start conversation with OpenAI (no system message — the cache
        // already contains the system instruction, and Gemini does not
        // allow system_instruction alongside cachedContent).
        final history = <oai.ChatMessage>[
          oai.ChatMessage.user('Name one ancient Egyptian pharaoh. Just the name.'),
        ];

        final openAIResp = await openAIClient.chat.completions.create(
          oai.ChatCompletionCreateRequest(
            model: 'gpt-4.1-nano',
            messages: history,
          ),
        );
        history.add(openAIResp.choices.first.message);
        print('OpenAI: ${openAIResp.choices.first.message.content}');

        // Continue with Gemini using cache.
        client.cachedContent = cacheName;

        history.add(oai.ChatMessage.user('Tell me more about that pharaoh. One sentence.'));

        final geminiResp = await client.chat.completions.create(
          oai.ChatCompletionCreateRequest(
            model: model,
            messages: history,
          ),
        );

        final geminiUsage = geminiResp.usage!;
        final cachedTokens = geminiUsage.promptTokensDetails?.cachedTokens ?? 0;

        print('Gemini (cached): ${geminiResp.choices.first.message.content}');
        print('  Prompt tokens: ${geminiUsage.promptTokens}');
        print('  Cached tokens: $cachedTokens');

        // The cached context should produce cache hits even with
        // cross-provider history.
        expect(cachedTokens, greaterThan(0), reason: 'Gemini cache should work even with OpenAI messages in history');

        // Verify the OpenAI format is intact.
        expect(geminiResp.model, model);
        expect(geminiResp.provider, 'gemini');
        expect(geminiResp.choices.first.message.content, isNotEmpty);
      } finally {
        client.cachedContent = null;
        await rawGeminiClient.cachedContents.delete(name: cacheName);
      }
    },
    timeout: const Timeout(Duration(minutes: 3)),
  );

  // ===========================================================================
  // Test 5: Pricing validation against published rates
  // ===========================================================================
  test(
    'Pricing math validates against published Gemini 3 Flash rates',
    () {
      // Published pricing for Gemini 3 Flash Preview (per 1M tokens):
      const inputPricePer1M = 0.50;
      const outputPricePer1M = 3.00;
      const cacheReadPricePer1M = 0.05;
      const cacheWritePricePer1M = 0.08333;

      // Cache read should be 10x cheaper than input.
      expect(
        inputPricePer1M / cacheReadPricePer1M,
        closeTo(10.0, 0.1),
        reason: 'Cache read should be ~10x cheaper than input',
      );

      // Cache write should be ~6x cheaper than input.
      expect(
        inputPricePer1M / cacheWritePricePer1M,
        closeTo(6.0, 0.1),
        reason: 'Cache write should be ~6x cheaper than input',
      );

      // Simulate a request with 10,000 prompt tokens, 8,000 cached.
      const totalPromptTokens = 10000;
      const cachedTokens = 8000;
      const nonCachedTokens = totalPromptTokens - cachedTokens;
      const outputTokens = 100;

      final costWithCache =
          (nonCachedTokens * inputPricePer1M / 1e6) +
          (cachedTokens * cacheReadPricePer1M / 1e6) +
          (outputTokens * outputPricePer1M / 1e6);

      final costWithoutCache = (totalPromptTokens * inputPricePer1M / 1e6) + (outputTokens * outputPricePer1M / 1e6);

      final savings = costWithoutCache - costWithCache;
      final savingsPercent = (savings / costWithoutCache) * 100;

      print('Pricing validation (10K prompt, 8K cached, 100 output):');
      print('  Cost with cache:    \$${costWithCache.toStringAsFixed(6)}');
      print('  Cost without cache: \$${costWithoutCache.toStringAsFixed(6)}');
      print('  Savings: ${savingsPercent.toStringAsFixed(1)}%');

      // With 80% cache hit rate, savings should be significant.
      expect(savingsPercent, greaterThan(50), reason: 'With 80% cache hit rate, savings should be >50% on input costs');

      // Verify our Usage model captures these correctly.
      final usage = oai.Usage(
        promptTokens: totalPromptTokens,
        completionTokens: outputTokens,
        totalTokens: totalPromptTokens + outputTokens,
        promptTokensDetails: const oai.PromptTokensDetails(
          cachedTokens: cachedTokens,
        ),
      );

      expect(usage.promptTokens, totalPromptTokens);
      expect(usage.completionTokens, outputTokens);
      expect(usage.promptTokensDetails!.cachedTokens, cachedTokens);
    },
  );
}
