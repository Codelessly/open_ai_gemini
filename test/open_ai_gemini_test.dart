import 'dart:convert';

import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:test/test.dart';

void main() {
  group('FinishReasonMapper', () {
    test('maps STOP to stop', () {
      expect(
        FinishReasonMapper.toOpenAI(gai.FinishReason.stop),
        oai.FinishReason.stop,
      );
    });

    test('maps MAX_TOKENS to length', () {
      expect(
        FinishReasonMapper.toOpenAI(gai.FinishReason.maxTokens),
        oai.FinishReason.length,
      );
    });

    test('maps safety reasons to content_filter', () {
      for (final reason in [
        gai.FinishReason.safety,
        gai.FinishReason.recitation,
        gai.FinishReason.blocklist,
        gai.FinishReason.prohibitedContent,
        gai.FinishReason.spii,
      ]) {
        expect(
          FinishReasonMapper.toOpenAI(reason),
          oai.FinishReason.contentFilter,
        );
      }
    });

    test('returns tool_calls when hasToolCalls is true', () {
      expect(
        FinishReasonMapper.toOpenAI(
          gai.FinishReason.stop,
          hasToolCalls: true,
        ),
        oai.FinishReason.toolCalls,
      );
    });
  });

  group('ToolMapper', () {
    test('converts OpenAI tools to Gemini function declarations', () {
      final tools = [
        oai.Tool.function(
          name: 'get_weather',
          description: 'Get weather for a location',
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

      final geminiTool = ToolMapper.toGeminiTools(tools);
      expect(geminiTool, isNotNull);
      expect(geminiTool!.functionDeclarations, hasLength(1));
      expect(geminiTool.functionDeclarations!.first.name, 'get_weather');
    });

    test('converts tool choice auto', () {
      final config = ToolMapper.toGeminiToolConfig(oai.ToolChoice.auto());
      expect(config, isNotNull);
      expect(
        config!.functionCallingConfig!.mode,
        gai.FunctionCallingMode.auto,
      );
    });

    test('converts tool choice required to ANY', () {
      final config = ToolMapper.toGeminiToolConfig(oai.ToolChoice.required());
      expect(
        config!.functionCallingConfig!.mode,
        gai.FunctionCallingMode.any,
      );
    });

    test('converts tool choice none', () {
      final config = ToolMapper.toGeminiToolConfig(oai.ToolChoice.none());
      expect(
        config!.functionCallingConfig!.mode,
        gai.FunctionCallingMode.none,
      );
    });

    test('converts named tool choice', () {
      final config = ToolMapper.toGeminiToolConfig(
        oai.ToolChoice.function('get_weather'),
      );
      expect(
        config!.functionCallingConfig!.mode,
        gai.FunctionCallingMode.any,
      );
      expect(
        config.functionCallingConfig!.allowedFunctionNames,
        ['get_weather'],
      );
    });

    test('sanitizes anyOf with const patterns to enum', () {
      final schema = ToolMapper.sanitizeSchema({
        'type': 'object',
        'properties': {
          'format': {
            'anyOf': [
              {'const': 'json'},
              {'const': 'markdown'},
            ],
          },
        },
      });

      final format = (schema['properties'] as Map<String, dynamic>)['format'] as Map<String, dynamic>;
      expect(format['enum'], ['json', 'markdown']);
      expect(format.containsKey('anyOf'), isFalse);
    });

    test('sanitizes anyOf with object variants by merging properties', () {
      final schema = ToolMapper.sanitizeSchema({
        'anyOf': [
          {
            'properties': {
              'name': {'type': 'string'},
            },
          },
          {
            'properties': {
              'age': {'type': 'integer'},
            },
          },
        ],
      });

      final props = schema['properties'] as Map<String, dynamic>;
      expect(props.containsKey('name'), isTrue);
      expect(props.containsKey('age'), isTrue);
    });

    test('strips OpenAI-specific fields', () {
      final schema = ToolMapper.sanitizeSchema({
        'type': 'object',
        'strict': true,
        'additionalProperties': false,
        'properties': {
          'name': {'type': 'string'},
        },
      });

      expect(schema.containsKey('strict'), isFalse);
      expect(schema.containsKey('additionalProperties'), isFalse);
    });
  });

  group('MessageContentConverter', () {
    test('extracts system messages as systemInstruction', () {
      final messages = [
        oai.ChatMessage.system('You are a helpful assistant.'),
        oai.ChatMessage.user('Hello!'),
      ];

      final result = MessageContentConverter.toGemini(messages);
      expect(result.systemInstruction, isNotNull);
      expect(
        (result.systemInstruction!.parts.first as gai.TextPart).text,
        'You are a helpful assistant.',
      );
      expect(result.contents, hasLength(1));
      expect(result.contents.first.role, 'user');
    });

    test('converts user text message', () {
      final messages = [oai.ChatMessage.user('Hello!')];
      final result = MessageContentConverter.toGemini(messages);

      expect(result.contents, hasLength(1));
      expect(result.contents.first.role, 'user');
      expect(
        (result.contents.first.parts.first as gai.TextPart).text,
        'Hello!',
      );
    });

    test('converts assistant message with content', () {
      final messages = [
        const oai.AssistantMessage(content: 'Hi there!'),
      ];
      final result = MessageContentConverter.toGemini(messages);

      expect(result.contents, hasLength(1));
      expect(result.contents.first.role, 'model');
      expect(
        (result.contents.first.parts.first as gai.TextPart).text,
        'Hi there!',
      );
    });

    test('converts assistant message with tool calls', () {
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          content: null,
          toolCalls: [
            oai.ToolCall(
              id: 'call_1',
              type: 'function',
              function: oai.FunctionCall(
                name: 'get_weather',
                arguments: '{"location":"SF"}',
              ),
            ),
          ],
        ),
      ];
      final result = MessageContentConverter.toGemini(messages);

      final parts = result.contents.first.parts;
      // Empty text part + function call part.
      expect(parts.whereType<gai.FunctionCallPart>(), hasLength(1));
      final fcPart = parts.whereType<gai.FunctionCallPart>().first;
      expect(fcPart.functionCall.name, 'get_weather');
      expect(fcPart.functionCall.args, {'location': 'SF'});
    });

    test('converts tool messages to function responses', () {
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          toolCalls: [
            oai.ToolCall(
              id: 'call_1',
              type: 'function',
              function: oai.FunctionCall(
                name: 'get_weather',
                arguments: '{"location":"SF"}',
              ),
            ),
          ],
        ),
        oai.ChatMessage.tool(
          toolCallId: 'call_1',
          content: '{"temp": 72}',
        ),
      ];
      final result = MessageContentConverter.toGemini(messages);

      // Assistant + user (tool response).
      expect(result.contents, hasLength(2));
      final toolContent = result.contents[1];
      expect(toolContent.role, 'user');
      expect(toolContent.parts.first, isA<gai.FunctionResponsePart>());
      final frPart = toolContent.parts.first as gai.FunctionResponsePart;
      expect(frPart.functionResponse.name, 'get_weather');
    });

    test('preserves thought signatures in round-trip', () {
      final signature = base64Encode([1, 2, 3, 4]);
      final thoughtSignatures = {'call_1': signature};

      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          toolCalls: [
            oai.ToolCall(
              id: 'call_1',
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
        thoughtSignatures: thoughtSignatures,
        sourceProvider: 'gemini',
        sourceModel: 'gemini-3-flash-preview',
        modelId: 'gemini-3-flash-preview',
      );

      final fcPart = result.contents.first.parts.whereType<gai.FunctionCallPart>().first;
      expect(fcPart.thoughtSignature, [1, 2, 3, 4]);
    });

    test('merges consecutive tool messages into single user content', () {
      final messages = <oai.ChatMessage>[
        const oai.AssistantMessage(
          toolCalls: [
            oai.ToolCall(
              id: 'call_1',
              type: 'function',
              function: oai.FunctionCall(name: 'fn1', arguments: '{}'),
            ),
            oai.ToolCall(
              id: 'call_2',
              type: 'function',
              function: oai.FunctionCall(name: 'fn2', arguments: '{}'),
            ),
          ],
        ),
        oai.ChatMessage.tool(toolCallId: 'call_1', content: 'result1'),
        oai.ChatMessage.tool(toolCallId: 'call_2', content: 'result2'),
      ];

      final result = MessageContentConverter.toGemini(messages);

      // Assistant + single user content with 2 function responses.
      expect(result.contents, hasLength(2));
      expect(
        result.contents[1].parts.whereType<gai.FunctionResponsePart>(),
        hasLength(2),
      );
    });
  });

  group('ChatCompletionRequestConverter', () {
    test('converts a basic request', () {
      final request = oai.ChatCompletionCreateRequest(
        model: 'gemini-2.5-pro',
        messages: [
          oai.ChatMessage.system('Be helpful'),
          oai.ChatMessage.user('Hello'),
        ],
        temperature: 0.7,
        maxTokens: 1000,
      );

      final result = ChatCompletionRequestConverter.convert(request);

      expect(result.systemInstruction, isNotNull);
      expect(result.contents, hasLength(1));
      expect(result.generationConfig, isNotNull);
      expect(result.generationConfig!.temperature, 0.7);
      expect(result.generationConfig!.maxOutputTokens, 1000);
    });

    test('converts reasoning effort to thinking config', () {
      final request = oai.ChatCompletionCreateRequest(
        model: 'gemini-2.5-pro',
        messages: [oai.ChatMessage.user('Think hard')],
        reasoningEffort: oai.ReasoningEffort.high,
      );

      final result = ChatCompletionRequestConverter.convert(request);

      expect(result.generationConfig!.thinkingConfig, isNotNull);
      expect(
        result.generationConfig!.thinkingConfig!.thinkingLevel,
        gai.ThinkingLevel.high,
      );
      expect(
        result.generationConfig!.thinkingConfig!.includeThoughts,
        isTrue,
      );
    });

    test('converts JSON response format', () {
      final request = oai.ChatCompletionCreateRequest(
        model: 'gemini-2.5-pro',
        messages: [oai.ChatMessage.user('Give JSON')],
        responseFormat: oai.ResponseFormat.jsonObject(),
      );

      final result = ChatCompletionRequestConverter.convert(request);
      expect(result.generationConfig!.responseMimeType, 'application/json');
    });

    test('converts tools and tool choice', () {
      final request = oai.ChatCompletionCreateRequest(
        model: 'gemini-2.5-pro',
        messages: [oai.ChatMessage.user('Get weather')],
        tools: [
          oai.Tool.function(
            name: 'get_weather',
            description: 'Get the weather',
            parameters: {
              'type': 'object',
              'properties': {
                'city': {'type': 'string'},
              },
            },
          ),
        ],
        toolChoice: oai.ToolChoice.auto(),
      );

      final result = ChatCompletionRequestConverter.convert(request);

      expect(result.tools, isNotNull);
      expect(result.tools, hasLength(1));
      expect(result.toolConfig, isNotNull);
    });
  });

  group('ChatCompletionResponseConverter', () {
    test('converts a text response', () {
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
          promptTokenCount: 10,
          candidatesTokenCount: 5,
          totalTokenCount: 15,
        ),
      );

      final result = ChatCompletionResponseConverter.convert(
        response,
        model: 'gemini-2.5-pro',
      );

      expect(result.completion.choices.first.message.content, 'Hello!');
      expect(
        result.completion.choices.first.finishReason,
        oai.FinishReason.stop,
      );
      expect(result.completion.usage!.promptTokens, 10);
      expect(result.completion.usage!.completionTokens, 5);
      expect(result.completion.provider, 'gemini');
    });

    test('converts a response with tool calls', () {
      final response = gai.GenerateContentResponse(
        candidates: [
          gai.Candidate(
            content: gai.Content(
              role: 'model',
              parts: [
                gai.FunctionCallPart(
                  gai.FunctionCall(
                    name: 'get_weather',
                    args: {'location': 'SF'},
                  ),
                ),
              ],
            ),
            finishReason: gai.FinishReason.stop,
          ),
        ],
      );

      final result = ChatCompletionResponseConverter.convert(
        response,
        model: 'gemini-2.5-pro',
      );

      final toolCalls = result.completion.choices.first.message.toolCalls;
      expect(toolCalls, isNotNull);
      expect(toolCalls, hasLength(1));
      expect(toolCalls!.first.function.name, 'get_weather');
      expect(
        jsonDecode(toolCalls.first.function.arguments),
        {'location': 'SF'},
      );
      expect(
        result.completion.choices.first.finishReason,
        oai.FinishReason.toolCalls,
      );
    });

    test('extracts thought signatures from function calls', () {
      final response = gai.GenerateContentResponse(
        candidates: [
          gai.Candidate(
            content: gai.Content(
              role: 'model',
              parts: [
                gai.FunctionCallPart(
                  gai.FunctionCall(
                    name: 'get_weather',
                    args: {},
                  ),
                  thoughtSignature: [10, 20, 30],
                ),
              ],
            ),
            finishReason: gai.FinishReason.stop,
          ),
        ],
      );

      final result = ChatCompletionResponseConverter.convert(
        response,
        model: 'gemini-2.5-pro',
      );

      expect(result.thoughtSignatures, isNotEmpty);
      final toolCallId = result.completion.choices.first.message.toolCalls!.first.id;
      expect(result.thoughtSignatures[toolCallId], isNotNull);
      // Verify round-trip.
      expect(
        base64Decode(result.thoughtSignatures[toolCallId]!),
        [10, 20, 30],
      );
    });

    test('handles thinking/reasoning parts', () {
      final response = gai.GenerateContentResponse(
        candidates: [
          gai.Candidate(
            content: gai.Content(
              role: 'model',
              parts: [
                gai.TextPart('Let me think...', thought: true),
                gai.TextPart('The answer is 42.'),
              ],
            ),
            finishReason: gai.FinishReason.stop,
          ),
        ],
      );

      final result = ChatCompletionResponseConverter.convert(
        response,
        model: 'gemini-2.5-pro',
      );

      expect(result.completion.choices.first.message.content, 'The answer is 42.');
      expect(
        result.completion.choices.first.message.reasoningContent,
        'Let me think...',
      );
    });

    test('handles empty candidates', () {
      final response = gai.GenerateContentResponse(candidates: []);

      final result = ChatCompletionResponseConverter.convert(
        response,
        model: 'gemini-2.5-pro',
      );

      expect(result.completion.choices.first.message.content, '');
    });
  });

  group('Gemini → OpenAI message conversion', () {
    test('converts model content to assistant message', () {
      final content = gai.Content(
        role: 'model',
        parts: [gai.TextPart('Hello!')],
      );

      final result = MessageContentConverter.toOpenAI(content);

      expect(result.messages, hasLength(1));
      final msg = result.messages.first as oai.AssistantMessage;
      expect(msg.content, 'Hello!');
    });

    test('converts user content with function responses to tool messages', () {
      final content = gai.Content(
        role: 'user',
        parts: [
          gai.FunctionResponsePart(
            gai.FunctionResponse(
              name: 'get_weather',
              response: {'temp': 72},
            ),
          ),
        ],
      );

      final result = MessageContentConverter.toOpenAI(content);

      expect(result.messages, hasLength(1));
      final msg = result.messages.first as oai.ToolMessage;
      expect(msg.toolCallId, 'get_weather');
    });

    test('extracts thought signatures from model content', () {
      final content = gai.Content(
        role: 'model',
        parts: [
          gai.FunctionCallPart(
            gai.FunctionCall(name: 'fn', args: {}),
            thoughtSignature: [5, 6, 7],
          ),
        ],
      );

      final result = MessageContentConverter.toOpenAI(content);

      expect(result.thoughtSignatures, isNotNull);
      expect(result.thoughtSignatures!.values.first, base64Encode([5, 6, 7]));
    });
  });
}
