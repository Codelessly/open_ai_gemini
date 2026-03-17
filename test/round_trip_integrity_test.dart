import 'dart:convert';

import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:test/test.dart';

/// Tests for OpenAI round-trip integrity.
/// Ensures that conversations stored in OpenAI format survive
/// conversion to Gemini and back without data loss.
void main() {
  group('OpenAI round-trip integrity', () {
    test('full conversation round-trips through Gemini conversion intact', () {
      final originalMessages = <oai.ChatMessage>[
        oai.ChatMessage.system('You are a helpful assistant.'),
        oai.ChatMessage.user('What is the weather in SF?'),
        const oai.AssistantMessage(
          toolCalls: [
            oai.ToolCall(
              id: 'call_0',
              type: 'function',
              function: oai.FunctionCall(
                name: 'get_weather',
                arguments: '{"location":"San Francisco"}',
              ),
            ),
          ],
        ),
        oai.ChatMessage.tool(
          toolCallId: 'call_0',
          content: '{"temp": 72, "condition": "sunny"}',
        ),
        const oai.AssistantMessage(
          content: 'The weather in SF is 72°F and sunny!',
        ),
        oai.ChatMessage.user('Thanks! What about in Tokyo?'),
      ];

      final geminiResult = MessageContentConverter.toGemini(originalMessages);

      expect(geminiResult.systemInstruction, isNotNull);
      expect(geminiResult.contents, isNotEmpty);

      final recoveredMessages = <oai.ChatMessage>[];
      if (geminiResult.systemInstruction != null) {
        final sysParts = geminiResult.systemInstruction!.parts;
        final sysText = sysParts.whereType<gai.TextPart>().map((p) => p.text).join('\n\n');
        recoveredMessages.add(oai.ChatMessage.system(sysText));
      }
      for (final content in geminiResult.contents) {
        final convResult = MessageContentConverter.toOpenAI(content);
        recoveredMessages.addAll(convResult.messages);
      }

      final originalSys = originalMessages.first as oai.SystemMessage;
      final recoveredSys = recoveredMessages.first as oai.SystemMessage;
      expect(recoveredSys.content, originalSys.content);

      final originalUsers = originalMessages.whereType<oai.UserMessage>().toList();
      final recoveredUsers = recoveredMessages.whereType<oai.UserMessage>().toList();
      expect(recoveredUsers.length, originalUsers.length);

      final originalAssistants = originalMessages.whereType<oai.AssistantMessage>().toList();
      final recoveredAssistants = recoveredMessages.whereType<oai.AssistantMessage>().toList();
      expect(recoveredAssistants.length, originalAssistants.length);

      expect(recoveredAssistants.last.content, originalAssistants.last.content);
    });

    test('tool call arguments survive round-trip exactly', () {
      final originalArgs = '{"location":"San Francisco","unit":"celsius"}';
      final messages = <oai.ChatMessage>[
        oai.AssistantMessage(
          toolCalls: [
            oai.ToolCall(
              id: 'call_0',
              type: 'function',
              function: oai.FunctionCall(
                name: 'get_weather',
                arguments: originalArgs,
              ),
            ),
          ],
        ),
      ];

      final geminiResult = MessageContentConverter.toGemini(messages);
      final openAIResult = MessageContentConverter.toOpenAI(geminiResult.contents.first);
      final recovered = openAIResult.messages.first as oai.AssistantMessage;

      final recoveredArgs = jsonDecode(recovered.toolCalls!.first.function.arguments);
      final originalParsed = jsonDecode(originalArgs);
      expect(recoveredArgs, originalParsed);
    });

    test('usage metadata with cached tokens round-trips correctly', () {
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
          promptTokenCount: 1000,
          candidatesTokenCount: 50,
          totalTokenCount: 1050,
          cachedContentTokenCount: 800,
        ),
      );

      final result = ChatCompletionResponseConverter.convert(
        response,
        model: 'gemini-3-flash-preview',
      );

      final usage = result.completion.usage!;
      expect(usage.promptTokens, 1000);
      expect(usage.completionTokens, 50);
      expect(usage.totalTokens, 1050);
      expect(usage.promptTokensDetails, isNotNull);
      expect(usage.promptTokensDetails!.cachedTokens, 800);
    });
  });
}
