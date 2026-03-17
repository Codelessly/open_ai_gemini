import 'dart:convert';

import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:test/test.dart';

/// Tests for thought signature handling:
/// - SKIP_THOUGHT_SIGNATURE sentinel for Gemini 3 (Gap #1)
/// - Text signature round-tripping (Gap #5)
/// - Thought signature validation (Gap #6)
void main() {
  group('SKIP_THOUGHT_SIGNATURE sentinel (Gap #1)', () {
    test('unsigned function calls on Gemini 3 get sentinel signature', () {
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          toolCalls: [
            oai.ToolCall(
              id: 'call_0',
              type: 'function',
              function: oai.FunctionCall(
                name: 'get_weather',
                arguments: '{"location":"SF"}',
              ),
            ),
          ],
        ),
      ];

      final result = MessageContentConverter.toGemini(
        messages,
        thoughtSignatures: {},
        modelId: 'gemini-3-flash-preview',
      );

      final fcPart = result.contents.first.parts.whereType<gai.FunctionCallPart>().first;

      expect(fcPart.thoughtSignature, isNotNull, reason: 'Gemini 3 requires thoughtSignature on all function calls');
      final sentinelString = utf8.decode(fcPart.thoughtSignature!);
      expect(sentinelString, 'skip_thought_signature_validator');
    });

    test('unsigned function calls on non-Gemini-3 models get NO sentinel', () {
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          toolCalls: [
            oai.ToolCall(
              id: 'call_0',
              type: 'function',
              function: oai.FunctionCall(
                name: 'get_weather',
                arguments: '{}',
              ),
            ),
          ],
        ),
      ];

      final result = MessageContentConverter.toGemini(
        messages,
        thoughtSignatures: {},
        modelId: 'gemini-2.5-flash',
      );

      final fcPart = result.contents.first.parts.whereType<gai.FunctionCallPart>().first;
      expect(fcPart.thoughtSignature, isNull);
    });

    test('signed function calls keep their real signature on Gemini 3', () {
      final realSignature = base64Encode([10, 20, 30, 40]);
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          toolCalls: [
            oai.ToolCall(
              id: 'call_0',
              type: 'function',
              function: oai.FunctionCall(
                name: 'get_weather',
                arguments: '{}',
              ),
            ),
          ],
        ),
      ];

      final result = MessageContentConverter.toGemini(
        messages,
        thoughtSignatures: {'call_0': realSignature},
        modelId: 'gemini-3-flash-preview',
        sourceProvider: 'gemini',
        sourceModel: 'gemini-3-flash-preview',
      );

      final fcPart = result.contents.first.parts.whereType<gai.FunctionCallPart>().first;
      expect(fcPart.thoughtSignature, [10, 20, 30, 40]);
    });
  });

  group('Text signature round-tripping (Gap #5)', () {
    test('__last_text__ signature is re-injected into text parts', () {
      final textSigBase64 = base64Encode([50, 60, 70]);
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(content: 'The answer is 42.'),
      ];

      final result = MessageContentConverter.toGemini(
        messages,
        thoughtSignatures: {'__last_text__': textSigBase64},
        modelId: 'gemini-3-flash-preview',
        sourceProvider: 'gemini',
        sourceModel: 'gemini-3-flash-preview',
      );

      final textPart = result.contents.first.parts.whereType<gai.TextPart>().where((p) => p.thought != true).first;

      expect(textPart.thoughtSignature, [
        50,
        60,
        70,
      ], reason: '__last_text__ signature should be injected into text parts');
    });

    test('text signature not injected for different model', () {
      final textSigBase64 = base64Encode([50, 60, 70]);
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(content: 'The answer is 42.'),
      ];

      final result = MessageContentConverter.toGemini(
        messages,
        thoughtSignatures: {'__last_text__': textSigBase64},
        modelId: 'gemini-3-flash-preview',
        sourceProvider: 'openai',
        sourceModel: 'gpt-5',
      );

      final textPart = result.contents.first.parts.whereType<gai.TextPart>().where((p) => p.thought != true).first;

      expect(textPart.thoughtSignature, isNull);
    });
  });

  group('Thought signature validation (Gap #6)', () {
    test('valid base64 signatures are preserved', () {
      final validSig = base64Encode([1, 2, 3, 4, 5, 6]);
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          toolCalls: [
            oai.ToolCall(
              id: 'call_0',
              type: 'function',
              function: oai.FunctionCall(name: 'fn', arguments: '{}'),
            ),
          ],
        ),
      ];

      final result = MessageContentConverter.toGemini(
        messages,
        thoughtSignatures: {'call_0': validSig},
        modelId: 'gemini-3-flash-preview',
        sourceProvider: 'gemini',
        sourceModel: 'gemini-3-flash-preview',
      );

      final fcPart = result.contents.first.parts.whereType<gai.FunctionCallPart>().first;
      expect(fcPart.thoughtSignature, isNotNull, reason: 'Valid base64 signatures should be preserved');
    });

    test('invalid (non-base64) signatures are rejected', () {
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          toolCalls: [
            oai.ToolCall(
              id: 'call_0',
              type: 'function',
              function: oai.FunctionCall(name: 'fn', arguments: '{}'),
            ),
          ],
        ),
      ];

      final result = MessageContentConverter.toGemini(
        messages,
        thoughtSignatures: {'call_0': 'not!valid!base64!!!'},
        modelId: 'gemini-3-flash-preview',
        sourceProvider: 'gemini',
        sourceModel: 'gemini-3-flash-preview',
      );

      final fcPart = result.contents.first.parts.whereType<gai.FunctionCallPart>().first;
      final sigString = utf8.decode(fcPart.thoughtSignature!);
      expect(
        sigString,
        'skip_thought_signature_validator',
        reason: 'Invalid signatures on Gemini 3 should use sentinel fallback',
      );
    });

    test('signatures from different provider are rejected', () {
      final validSig = base64Encode([1, 2, 3]);
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          toolCalls: [
            oai.ToolCall(
              id: 'call_0',
              type: 'function',
              function: oai.FunctionCall(name: 'fn', arguments: '{}'),
            ),
          ],
        ),
      ];

      final result = MessageContentConverter.toGemini(
        messages,
        thoughtSignatures: {'call_0': validSig},
        modelId: 'gemini-3-flash-preview',
        sourceProvider: 'openai',
        sourceModel: 'gpt-5',
      );

      final fcPart = result.contents.first.parts.whereType<gai.FunctionCallPart>().first;
      final sigString = utf8.decode(fcPart.thoughtSignature!);
      expect(sigString, 'skip_thought_signature_validator');
    });

    group('isValidThoughtSignature utility', () {
      test('accepts valid base64', () {
        expect(isValidThoughtSignature(base64Encode([1, 2, 3])), isTrue);
        expect(isValidThoughtSignature('AQID'), isTrue); // [1,2,3]
      });

      test('rejects null and empty', () {
        expect(isValidThoughtSignature(null), isFalse);
        expect(isValidThoughtSignature(''), isFalse);
      });

      test('rejects non-base64 strings', () {
        expect(isValidThoughtSignature('not!valid!'), isFalse);
        expect(isValidThoughtSignature('abc'), isFalse); // not multiple of 4
      });
    });

    group('isGemini3Model utility', () {
      test('detects Gemini 3 models', () {
        expect(isGemini3Model('gemini-3-flash-preview'), isTrue);
        expect(isGemini3Model('gemini-3-pro'), isTrue);
        expect(isGemini3Model('gemini-3.5-flash'), isTrue);
      });

      test('rejects non-Gemini-3 models', () {
        expect(isGemini3Model('gemini-2.5-flash'), isFalse);
        expect(isGemini3Model('gemini-2.5-pro'), isFalse);
        expect(isGemini3Model(null), isFalse);
      });
    });
  });
}
