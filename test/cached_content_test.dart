import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:test/test.dart';

void main() {
  group('GeminiOpenAIClient cached content', () {
    late GeminiOpenAIClient client;

    setUp(() {
      client = GeminiOpenAIClient(apiKey: 'test-key');
    });

    tearDown(() {
      client.close();
    });

    test('exposes cachedContent property and cachedContents resource', () {
      expect(client.cachedContent, isNull);
      expect(client.cachedContents, isNotNull);

      client.cachedContent = 'cachedContents/abc123';
      expect(client.cachedContent, 'cachedContents/abc123');
    });

    test('clearConversationState does not clear cachedContent', () {
      client.cachedContent = 'cachedContents/abc123';
      client.clearConversationState();
      expect(
        client.cachedContent,
        'cachedContents/abc123',
        reason: 'cached content is managed separately from conversation state',
      );
    });
  });

  group('ChatCompletionRequestConverter cached content pass-through', () {
    test('convert forwards cachedContent onto the result', () {
      final request = oai.ChatCompletionCreateRequest(
        model: 'gemini-2.5-flash',
        messages: [oai.ChatMessage.user('Hello')],
      );

      final withCache = ChatCompletionRequestConverter.convert(
        request,
        cachedContent: 'cachedContents/abc123',
      );
      final withoutCache = ChatCompletionRequestConverter.convert(request);

      expect(withCache.cachedContent, 'cachedContents/abc123');
      expect(withoutCache.cachedContent, isNull);
    });
  });

  group('ChatCompletionResponseConverter cached token reporting', () {
    test('reports cached token count in promptTokensDetails', () {
      final response = gai.GenerateContentResponse(
        candidates: [
          gai.Candidate(
            content: gai.Content(
              role: 'model',
              parts: [gai.TextPart('Hello!')],
            ),
            finishReason: gai.FinishReason.stop,
          ),
        ],
        usageMetadata: gai.UsageMetadata(
          promptTokenCount: 100,
          candidatesTokenCount: 50,
          totalTokenCount: 150,
          cachedContentTokenCount: 80,
        ),
      );

      final result = ChatCompletionResponseConverter.convert(
        response,
        model: 'gemini-2.5-flash',
      );

      expect(
        result.completion.usage!.promptTokensDetails!.cachedTokens,
        80,
      );
    });

    test('omits promptTokensDetails when no cached tokens', () {
      final response = gai.GenerateContentResponse(
        candidates: [
          gai.Candidate(
            content: gai.Content(
              role: 'model',
              parts: [gai.TextPart('Hello!')],
            ),
            finishReason: gai.FinishReason.stop,
          ),
        ],
        usageMetadata: gai.UsageMetadata(
          promptTokenCount: 100,
          candidatesTokenCount: 50,
          totalTokenCount: 150,
        ),
      );

      final result = ChatCompletionResponseConverter.convert(
        response,
        model: 'gemini-2.5-flash',
      );

      expect(result.completion.usage!.promptTokensDetails, isNull);
    });
  });
}
