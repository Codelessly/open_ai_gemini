import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:test/test.dart';

/// Tests for cross-provider/model awareness (Gap #2).
void main() {
  group('Cross-provider/model awareness (Gap #2)', () {
    test('reasoning from same model is kept as thought:true', () {
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          content: 'The answer is 42.',
          reasoningContent: 'Let me think about this...',
        ),
      ];

      final result = MessageContentConverter.toGemini(
        messages,
        sourceProvider: 'gemini',
        sourceModel: 'gemini-3-flash-preview',
        modelId: 'gemini-3-flash-preview',
      );

      final parts = result.contents.first.parts;
      final thoughtParts = parts.whereType<gai.TextPart>().where((p) => p.thought == true);
      expect(thoughtParts, isNotEmpty, reason: 'Same-model reasoning should be kept as thought:true');
      expect(thoughtParts.first.text, 'Let me think about this...');
    });

    test('reasoning from different model is converted to plain text', () {
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          content: 'The answer is 42.',
          reasoningContent: 'Let me think about this...',
        ),
      ];

      final result = MessageContentConverter.toGemini(
        messages,
        sourceProvider: 'openai',
        sourceModel: 'gpt-5',
        modelId: 'gemini-3-flash-preview',
      );

      final parts = result.contents.first.parts;
      final thoughtParts = parts.whereType<gai.TextPart>().where((p) => p.thought == true);

      expect(thoughtParts, isEmpty, reason: 'Different-provider reasoning should be plain text, not thought');

      final textParts = parts.whereType<gai.TextPart>();
      final allText = textParts.map((p) => p.text).join(' ');
      expect(allText, contains('Let me think about this...'));
    });

    test('reasoning without provider info defaults to plain text', () {
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          content: 'Hello.',
          reasoningContent: 'Internal reasoning...',
        ),
      ];

      final result = MessageContentConverter.toGemini(
        messages,
        modelId: 'gemini-3-flash-preview',
      );

      final parts = result.contents.first.parts;
      final thoughtParts = parts.whereType<gai.TextPart>().where((p) => p.thought == true);

      expect(thoughtParts, isEmpty);
    });
  });
}
