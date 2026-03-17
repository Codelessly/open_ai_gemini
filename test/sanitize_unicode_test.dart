import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:test/test.dart';

/// Tests for unicode surrogate sanitization (Gap #4) and
/// empty block filtering (Gap #10).
void main() {
  group('Unicode surrogate sanitization (Gap #4)', () {
    test('sanitizes lone surrogates in user text', () {
      final textWithSurrogate = 'Hello \uD800 World';
      final messages = [oai.ChatMessage.user(textWithSurrogate)];
      final result = MessageContentConverter.toGemini(messages);

      final textPart = result.contents.first.parts.first as gai.TextPart;
      expect(textPart.text.contains('\uD800'), isFalse, reason: 'Lone surrogates must be sanitized');
    });

    test('sanitizes surrogates in system instructions', () {
      final messages = [
        oai.ChatMessage.system('System \uD800 prompt'),
        oai.ChatMessage.user('Hi'),
      ];
      final result = MessageContentConverter.toGemini(messages);

      final sysText = (result.systemInstruction!.parts.first as gai.TextPart).text;
      expect(sysText.contains('\uD800'), isFalse);
    });

    test('sanitizes surrogates in assistant content', () {
      final messages = <oai.ChatMessage>[
        oai.AssistantMessage(content: 'Reply \uD800 here'),
      ];
      final result = MessageContentConverter.toGemini(messages);

      final textPart = result.contents.first.parts.first as gai.TextPart;
      expect(textPart.text.contains('\uD800'), isFalse);
    });

    test('clean text passes through unchanged', () {
      final messages = [oai.ChatMessage.user('Hello World! 123 ñ 日本語')];
      final result = MessageContentConverter.toGemini(messages);

      final textPart = result.contents.first.parts.first as gai.TextPart;
      expect(textPart.text, 'Hello World! 123 ñ 日本語');
    });

    group('sanitizeSurrogates utility', () {
      test('replaces lone high surrogate', () {
        expect(sanitizeSurrogates('a\uD800b'), 'a\uFFFDb');
      });

      test('replaces lone low surrogate', () {
        expect(sanitizeSurrogates('a\uDC00b'), 'a\uFFFDb');
      });

      test('preserves valid surrogate pairs', () {
        // 𝄞 (U+1D11E) is encoded as surrogate pair \uD834\uDD1E
        const music = '𝄞';
        expect(sanitizeSurrogates(music), music);
      });

      test('preserves ASCII', () {
        expect(sanitizeSurrogates('Hello, World!'), 'Hello, World!');
      });

      test('preserves BMP characters', () {
        expect(sanitizeSurrogates('café'), 'café');
        expect(sanitizeSurrogates('日本語'), '日本語');
      });
    });
  });

  group('Empty block filtering (Gap #10)', () {
    test('empty text content does not produce empty TextPart', () {
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          content: '',
          toolCalls: [
            oai.ToolCall(
              id: 'call_0',
              type: 'function',
              function: oai.FunctionCall(name: 'fn', arguments: '{}'),
            ),
          ],
        ),
      ];

      final result = MessageContentConverter.toGemini(messages);
      final parts = result.contents.first.parts;

      final textParts = parts.whereType<gai.TextPart>();
      for (final tp in textParts) {
        expect(tp.text.trim(), isNotEmpty, reason: 'Empty text parts should be filtered out');
      }
    });

    test('whitespace-only reasoning is filtered', () {
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          content: 'Real text.',
          reasoningContent: '   \n  ',
        ),
      ];

      final result = MessageContentConverter.toGemini(messages);
      final parts = result.contents.first.parts;

      final thoughtParts = parts.whereType<gai.TextPart>().where((p) => p.thought == true);
      expect(thoughtParts, isEmpty, reason: 'Whitespace-only reasoning should be filtered');
    });

    test('message with only empty content still produces valid Content', () {
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(content: null),
      ];

      final result = MessageContentConverter.toGemini(messages);
      expect(result.contents, hasLength(1));
      expect(result.contents.first.parts, isNotEmpty);
    });
  });
}
