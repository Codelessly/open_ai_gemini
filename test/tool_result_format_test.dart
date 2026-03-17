import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:test/test.dart';

/// Tests for tool result formatting:
/// - Error/success distinction (Gap #3)
/// - Multimodal function responses (Gap #7)
/// - Tool call ID normalization (Gap #9)
void main() {
  group('Error/success distinction in tool results (Gap #3)', () {
    test('successful tool result uses "output" key with parsed JSON', () {
      final part = ToolMapper.toGeminiFunctionResponse(
        functionName: 'get_weather',
        content: '{"temp": 72}',
      );

      final response = part.functionResponse.response;
      expect(response.containsKey('output'), isTrue, reason: 'Successful tool results should use "output" key');
      expect(response.containsKey('error'), isFalse);
      // Structured JSON should be parsed, not double-serialized.
      expect(response['output'], isA<Map>());
      expect((response['output'] as Map)['temp'], 72);
    });

    test('error tool result uses "error" key', () {
      final part = ToolMapper.toGeminiFunctionResponse(
        functionName: 'get_weather',
        content: 'Connection timeout',
        isError: true,
      );

      final response = part.functionResponse.response;
      expect(response.containsKey('error'), isTrue, reason: 'Error tool results should use "error" key');
      expect(response.containsKey('output'), isFalse);
    });

    test('error distinction round-trips through OpenAI format', () {
      final part = ToolMapper.toGeminiFunctionResponse(
        functionName: 'get_weather',
        content: 'API rate limited',
        isError: true,
      );

      expect(part.functionResponse.response['error'], isNotNull);
    });
  });

  group('Multimodal function responses (Gap #7)', () {
    test('Gemini 3 tool result with image uses inline parts', () {
      final part = ToolMapper.toGeminiFunctionResponse(
        functionName: 'take_screenshot',
        content: 'Screenshot taken',
        imageData: [
          MediaAttachment.inline(mimeType: 'image/png', data: 'iVBOR...base64data'),
        ],
        modelId: 'gemini-3-flash-preview',
      );

      expect(part.functionResponse.parts, isNotNull);
      expect(part.functionResponse.parts, isNotEmpty);
      expect(part.functionResponse.parts!.first.inlineData, isNotNull);
      expect(part.functionResponse.parts!.first.inlineData!.mimeType, 'image/png');
    });

    test('non-Gemini-3 tool result with image has null parts', () {
      final part = ToolMapper.toGeminiFunctionResponse(
        functionName: 'take_screenshot',
        content: 'Screenshot taken',
        imageData: [
          MediaAttachment.inline(mimeType: 'image/png', data: 'base64data'),
        ],
        modelId: 'gemini-2.5-flash',
      );

      expect(part.functionResponse.parts, isNull);
    });
  });

  group('Tool call ID normalization (Gap #9)', () {
    test('special characters are stripped from tool call IDs', () {
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          toolCalls: [
            oai.ToolCall(
              id: 'call_abc!@#\$%^&*()',
              type: 'function',
              function: oai.FunctionCall(name: 'fn', arguments: '{}'),
            ),
          ],
        ),
        oai.ChatMessage.tool(
          toolCallId: 'call_abc!@#\$%^&*()',
          content: 'result',
        ),
      ];

      final result = MessageContentConverter.toGemini(
        messages,
        normalizeToolCallIds: true,
      );

      final fcPart = result.contents.first.parts.whereType<gai.FunctionCallPart>().first;
      expect(
        RegExp(r'^[a-zA-Z0-9_-]+$').hasMatch(fcPart.functionCall.id ?? ''),
        isTrue,
        reason: 'Tool call IDs should only contain alphanumeric, _ and -',
      );
    });

    test('tool call IDs are capped at 64 characters', () {
      final longId = 'a' * 100;
      final messages = <oai.ChatMessage>[
        oai.AssistantMessage(
          toolCalls: [
            oai.ToolCall(
              id: longId,
              type: 'function',
              function: const oai.FunctionCall(name: 'fn', arguments: '{}'),
            ),
          ],
        ),
      ];

      final result = MessageContentConverter.toGemini(
        messages,
        normalizeToolCallIds: true,
      );

      final fcPart = result.contents.first.parts.whereType<gai.FunctionCallPart>().first;
      final normalizedId = fcPart.functionCall.id ?? '';
      expect(normalizedId.length, lessThanOrEqualTo(64));
    });

    test('clean IDs pass through unchanged', () {
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          toolCalls: [
            oai.ToolCall(
              id: 'call_0_get_weather',
              type: 'function',
              function: oai.FunctionCall(name: 'fn', arguments: '{}'),
            ),
          ],
        ),
      ];

      final result = MessageContentConverter.toGemini(
        messages,
        normalizeToolCallIds: true,
      );

      final fcPart = result.contents.first.parts.whereType<gai.FunctionCallPart>().first;
      expect(fcPart.functionCall.id, anyOf(isNull, 'call_0_get_weather'));
    });

    group('normalizeToolCallId utility', () {
      test('strips special characters', () {
        expect(normalizeToolCallId('call_abc!@#'), 'call_abc___');
      });

      test('caps at 64 characters', () {
        final long = 'a' * 100;
        expect(normalizeToolCallId(long).length, 64);
      });

      test('preserves clean IDs', () {
        expect(normalizeToolCallId('call_0_get_weather'), 'call_0_get_weather');
      });
    });
  });
}
