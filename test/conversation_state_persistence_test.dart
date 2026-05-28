import 'dart:async';
import 'dart:convert';

import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:test/test.dart';

/// Regression tests for the multi-turn Gemini 3 tool-call bug.
///
/// The bug shape: when an upstream wrapper (e.g. agent_kit) calls
/// `reset() + init(history: existingHistory)` between turns, the Gemini 3
/// follow-up request fails with HTTP 400:
///
///   "Function call is missing a thought_signature in functionCall parts."
///
/// Root cause: thought signatures used to live ONLY in an in-memory map on
/// the client. The map was either wiped by `clearConversationState()` (older
/// versions) or — after the first attempted fix — kept alive but useless,
/// because the JSON history rehydrated by the caller dropped all signatures
/// on the floor (the OpenAI `AssistantMessage` schema has no field for them).
///
/// Fix: thought signatures are now encoded directly into the
/// `tool_call.id` field as `tsig_<base64Url>__<originalId>`. The id is
/// opaque protocol metadata — the model never sees it — and it survives any
/// JSON round-trip through the caller's history store. The
/// [GeminiOpenAIClient.thoughtSignatures] map is back to being a hot
/// per-conversation cache and IS wiped by [GeminiOpenAIClient.clearConversationState].
///
/// These tests pin the fixed behaviour.
void main() {
  group('clearConversationState wipes signatures', () {
    late GeminiOpenAIClient client;

    setUp(() {
      client = GeminiOpenAIClient(apiKey: 'test-key');
    });

    tearDown(() {
      client.close();
    });

    test('clearConversationState wipes accumulated signatures', () {
      client.thoughtSignatures['call_hash_text_0'] = base64Encode([1, 2, 3, 4]);
      client.thoughtSignatures['__last_text__'] = base64Encode([9, 9, 9]);

      client.clearConversationState();

      expect(
        client.thoughtSignatures,
        isEmpty,
        reason:
            'clearConversationState must wipe the in-memory signature cache. '
            'Cross-conversation persistence is handled by tool_call.id '
            'encoding, not by this map.',
      );
    });

    test('clearThoughtSignatures also wipes (alias)', () {
      client.thoughtSignatures['call_x'] = base64Encode([7, 7, 7]);

      client.clearThoughtSignatures();

      expect(client.thoughtSignatures, isEmpty);
    });
  });

  group('tool_call.id encoding: response converter', () {
    test(
      'Gemini response with thought_signature → tool_call.id starts with `tsig_`',
      () {
        final signatureBytes = [10, 20, 30, 40, 50];
        final response = gai.GenerateContentResponse(
          responseId: 'rsp_1',
          candidates: [
            gai.Candidate(
              content: gai.Content(
                role: 'model',
                parts: [
                  gai.FunctionCallPart(
                    const gai.FunctionCall(
                      name: 'hash_text',
                      args: {'text': 'hello'},
                    ),
                    thoughtSignature: signatureBytes,
                  ),
                ],
              ),
            ),
          ],
        );

        final result = ChatCompletionResponseConverter.convert(
          response,
          model: 'gemini-3-flash-preview',
        );

        final toolCall = result.completion.choices.first.message.toolCalls!.first;
        expect(
          toolCall.id.startsWith('tsig_'),
          isTrue,
          reason: 'Encoded id must carry the `tsig_` prefix so downstream '
              'converters can detect and decode the embedded signature.',
        );

        final decoded = decodeThoughtSignatureFromToolCallId(toolCall.id);
        expect(decoded.signatureBase64, isNotNull);
        expect(base64Decode(decoded.signatureBase64!), signatureBytes);
        expect(decoded.originalId.startsWith('call_'), isTrue);
      },
    );

    test(
      'Encoded id survives toJson/fromJson round-trip through AssistantMessage',
      () {
        final signatureBytes = [99, 88, 77, 66];
        final response = gai.GenerateContentResponse(
          candidates: [
            gai.Candidate(
              content: gai.Content(
                role: 'model',
                parts: [
                  gai.FunctionCallPart(
                    const gai.FunctionCall(name: 'do_thing', args: {}),
                    thoughtSignature: signatureBytes,
                  ),
                ],
              ),
            ),
          ],
        );

        final result = ChatCompletionResponseConverter.convert(
          response,
          model: 'gemini-3-flash-preview',
        );
        final assistant = result.completion.choices.first.message;

        // Round-trip through JSON like a caller's history store would do.
        final json = assistant.toJson();
        final rehydrated = oai.AssistantMessage.fromJson(json);

        final rehydratedId = rehydrated.toolCalls!.first.id;
        expect(
          rehydratedId.startsWith('tsig_'),
          isTrue,
          reason: 'tool_call.id must survive JSON round-trip with prefix intact',
        );

        final decoded = decodeThoughtSignatureFromToolCallId(rehydratedId);
        expect(base64Decode(decoded.signatureBase64!), signatureBytes);
      },
    );

    test(
      'Gemini response WITHOUT thought_signature → plain tool_call.id (no tsig_ prefix)',
      () {
        final response = gai.GenerateContentResponse(
          candidates: [
            gai.Candidate(
              content: gai.Content(
                role: 'model',
                parts: [
                  gai.FunctionCallPart(
                    const gai.FunctionCall(name: 'do_thing', args: {}),
                    // no thoughtSignature
                  ),
                ],
              ),
            ),
          ],
        );

        final result = ChatCompletionResponseConverter.convert(
          response,
          model: 'gemini-2.5-flash',
        );

        final toolCall = result.completion.choices.first.message.toolCalls!.first;
        expect(toolCall.id.startsWith('tsig_'), isFalse);
      },
    );
  });

  group('tool_call.id encoding: outgoing converter (toGemini)', () {
    test(
      'AssistantMessage with `tsig_`-encoded id → FunctionCallPart carries decoded signature',
      () {
        final signatureBytes = [10, 20, 30, 40, 50];
        final encodedId = encodeThoughtSignatureInToolCallId(
          signatureBase64: base64Encode(signatureBytes),
          originalId: 'call_0_hash_text',
        );

        final messages = <oai.ChatMessage>[
          oai.ChatMessage.user('hash this'),
          oai.AssistantMessage(
            toolCalls: [
              oai.ToolCall(
                id: encodedId,
                type: 'function',
                function: const oai.FunctionCall(
                  name: 'hash_text',
                  arguments: '{"text":"hi"}',
                ),
              ),
            ],
          ),
        ];

        // Critically: NO thoughtSignatures map and NO source tags. The
        // encoded id alone is enough — that's the whole point of the design.
        final result = MessageContentConverter.toGemini(
          messages,
          modelId: 'gemini-3-flash-preview',
        );

        final fcPart = result.contents.last.parts.whereType<gai.FunctionCallPart>().first;

        expect(fcPart.thoughtSignature, isNotNull);
        expect(
          fcPart.thoughtSignature,
          signatureBytes,
          reason:
              'The decoded signature from the encoded id must end up on the '
              'outgoing FunctionCallPart verbatim — no map, no source tagging, '
              'just the id itself.',
        );
      },
    );

    test(
      'AssistantMessage without tsig_ prefix → falls back to in-memory map (legacy path)',
      () {
        const rawId = 'call_legacy_42';
        final signatureBytes = [7, 7, 7, 7];
        final signatureBase64 = base64Encode(signatureBytes);

        final messages = <oai.ChatMessage>[
          const oai.AssistantMessage(
            toolCalls: [
              oai.ToolCall(
                id: rawId,
                type: 'function',
                function: oai.FunctionCall(name: 'do_thing', arguments: '{}'),
              ),
            ],
          ),
        ];

        final result = MessageContentConverter.toGemini(
          messages,
          thoughtSignatures: {rawId: signatureBase64},
          modelId: 'gemini-3-flash-preview',
          // Same-provider/model tagging required for legacy path.
          sourceProvider: 'gemini',
          sourceModel: 'gemini-3-flash-preview',
        );

        final fcPart = result.contents.first.parts.whereType<gai.FunctionCallPart>().first;
        expect(fcPart.thoughtSignature, signatureBytes);
      },
    );

    test(
      'AssistantMessage without tsig_ prefix and no map entry → Gemini 3 sentinel fallback',
      () {
        final messages = <oai.ChatMessage>[
          const oai.AssistantMessage(
            toolCalls: [
              oai.ToolCall(
                id: 'call_unsigned',
                type: 'function',
                function: oai.FunctionCall(name: 'do_thing', arguments: '{}'),
              ),
            ],
          ),
        ];

        final result = MessageContentConverter.toGemini(
          messages,
          modelId: 'gemini-3-flash-preview',
        );

        final fcPart = result.contents.first.parts.whereType<gai.FunctionCallPart>().first;
        expect(
          utf8.decode(fcPart.thoughtSignature!),
          'skip_thought_signature_validator',
        );
      },
    );

    test(
      'Encoded id wins even when source tags are missing/mismatched (cross-provider replay)',
      () {
        // The whole appeal of id-encoded signatures: they are
        // self-authenticating. Even if a caller forgets to pass
        // sourceProvider/sourceModel, or replays the encoded id from a
        // different provider's history, the signature still flows through.
        final signatureBytes = [1, 2, 3];
        final encodedId = encodeThoughtSignatureInToolCallId(
          signatureBase64: base64Encode(signatureBytes),
          originalId: 'call_abc',
        );

        final messages = <oai.ChatMessage>[
          oai.AssistantMessage(
            toolCalls: [
              oai.ToolCall(
                id: encodedId,
                type: 'function',
                function: const oai.FunctionCall(name: 'fn', arguments: '{}'),
              ),
            ],
          ),
        ];

        final result = MessageContentConverter.toGemini(
          messages,
          modelId: 'gemini-3-flash-preview',
          // intentionally no source tags
        );

        final fcPart = result.contents.first.parts.whereType<gai.FunctionCallPart>().first;
        expect(fcPart.thoughtSignature, signatureBytes);
      },
    );
  });

  group('End-to-end: reset + rehydrate from JSON still preserves signatures', () {
    test('full round-trip: Gemini response → JSON history → next request', () {
      // Step 1: Gemini returns a function_call with a thought_signature.
      final signatureBytes = [42, 42, 42, 42];
      final response = gai.GenerateContentResponse(
        candidates: [
          gai.Candidate(
            content: gai.Content(
              role: 'model',
              parts: [
                gai.FunctionCallPart(
                  const gai.FunctionCall(
                    name: 'hash_text',
                    args: {'text': 'hello'},
                  ),
                  thoughtSignature: signatureBytes,
                ),
              ],
            ),
          ),
        ],
      );

      final converted = ChatCompletionResponseConverter.convert(
        response,
        model: 'gemini-3-flash-preview',
      );

      final assistantMessage = converted.completion.choices.first.message;

      // Step 2: Caller serializes assistant message and tool result to JSON.
      final assistantJson = assistantMessage.toJson();
      final toolJson = oai
          .ChatMessage.tool(
            toolCallId: assistantMessage.toolCalls!.first.id,
            content: '"5d41402abc4b2a76b9719d911017c592"',
          )
          .toJson();

      // Step 3: A NEW client (simulating reset() + init(history: ...))
      // rehydrates the history. No in-memory signature carryover.
      final freshClient = GeminiOpenAIClient(apiKey: 'test-key');
      addTearDown(freshClient.close);
      expect(freshClient.thoughtSignatures, isEmpty);

      final rehydratedHistory = <oai.ChatMessage>[
        oai.ChatMessage.user('hash hello'),
        oai.AssistantMessage.fromJson(assistantJson),
        oai.ChatMessage.fromJson(toolJson),
      ];

      // Step 4: Build the next outgoing Gemini request from the rehydrated
      // history. The signature MUST be on the FunctionCallPart for Gemini 3
      // to accept the request.
      final outgoing = MessageContentConverter.toGemini(
        rehydratedHistory,
        modelId: 'gemini-3-flash-preview',
        sourceProvider: 'gemini',
        sourceModel: 'gemini-3-flash-preview',
      );

      // The assistant content is the second item (after the user content).
      final assistantContent = outgoing.contents[1];
      final fcPart = assistantContent.parts.whereType<gai.FunctionCallPart>().first;
      expect(
        fcPart.thoughtSignature,
        signatureBytes,
        reason: 'After reset+rehydrate, the original signature must still '
            'reach the outgoing FunctionCallPart via the encoded id.',
      );
    });
  });

  group('Race-free stream consumption', () {
    test(
      'tool_call signatures are encoded in id immediately as events emit '
      '(no fire-and-forget completion dependency)',
      () async {
        // Feed a synthetic Gemini stream with a function_call that carries
        // a signature, and verify the resulting tool_call.id (the thing
        // downstream consumers see) carries the encoded signature at the
        // moment the event is emitted — i.e. without waiting on any
        // post-completion Future.
        final signatureBytes = [55, 66, 77];
        final geminiStream = Stream<gai.GenerateContentResponse>.fromIterable([
          gai.GenerateContentResponse(
            candidates: [
              gai.Candidate(
                content: gai.Content(
                  role: 'model',
                  parts: [
                    gai.FunctionCallPart(
                      const gai.FunctionCall(name: 'fn', args: {}),
                      thoughtSignature: signatureBytes,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ]);

        final result = convertGeminiStream(
          geminiStream,
          model: 'gemini-3-flash-preview',
        );

        // Collect tool_call ids from the stream synchronously as they arrive.
        final observedToolCallIds = <String>[];
        await for (final event in result.events) {
          final choices = event.choices;
          if (choices == null) continue;
          for (final choice in choices) {
            final toolCalls = choice.delta.toolCalls;
            if (toolCalls == null) continue;
            for (final tc in toolCalls) {
              final id = tc.id;
              if (id != null) observedToolCallIds.add(id);
            }
          }
        }

        // The tool_call id observed during stream consumption must already
        // carry the `tsig_` prefix. We do NOT await
        // `result.thoughtSignatures` before this assertion.
        expect(observedToolCallIds, hasLength(1));
        final encodedId = observedToolCallIds.single;
        expect(encodedId.startsWith('tsig_'), isTrue);

        final decoded = decodeThoughtSignatureFromToolCallId(encodedId);
        expect(base64Decode(decoded.signatureBase64!), signatureBytes);
      },
    );
  });

  group('Encode/decode helpers', () {
    test('encode → decode is lossless for the signature bytes', () {
      final signatureBytes = [
        0, 1, 2, 127, 128, 254, 255, // edge values
        for (var i = 0; i < 64; i++) i,
      ];
      final base64 = base64Encode(signatureBytes);

      final encodedId = encodeThoughtSignatureInToolCallId(
        signatureBase64: base64,
        originalId: 'call_0_fn',
      );

      expect(encodedId.startsWith('tsig_'), isTrue);
      expect(encodedId.endsWith('__call_0_fn'), isTrue);

      final decoded = decodeThoughtSignatureFromToolCallId(encodedId);
      expect(decoded.originalId, 'call_0_fn');
      expect(base64Decode(decoded.signatureBase64!), signatureBytes);
    });

    test('decode on a plain (non-prefixed) id returns input unchanged with null sig', () {
      final decoded = decodeThoughtSignatureFromToolCallId('call_abc_123');
      expect(decoded.signatureBase64, isNull);
      expect(decoded.originalId, 'call_abc_123');
    });

    test('decode on a malformed `tsig_` id (no separator) falls back to plain', () {
      final decoded = decodeThoughtSignatureFromToolCallId('tsig_garbage');
      expect(decoded.signatureBase64, isNull);
      expect(decoded.originalId, 'tsig_garbage');
    });
  });

  group('sanitizeMessagesForNonGeminiProvider', () {
    test(
      'strips `tsig_` prefix so AssistantMessage tool_call.id fits OpenAI 64-char cap',
      () {
        // A realistic Gemini 3 signature is ~50 bytes → ~70 base64-url chars,
        // pushing the encoded id well past 64 chars. Build one that mirrors
        // the production failure mode.
        final signatureBytes = List<int>.generate(50, (i) => i & 0xff);
        final encodedId = encodeThoughtSignatureInToolCallId(
          signatureBase64: base64Encode(signatureBytes),
          originalId: 'call_4',
        );
        expect(
          encodedId.length,
          greaterThan(64),
          reason:
              'Test fixture must reproduce the >64-char id condition that '
              'breaks OpenAI requests.',
        );

        final messages = <oai.ChatMessage>[
          oai.ChatMessage.user('hi'),
          oai.AssistantMessage(
            toolCalls: [
              oai.ToolCall(
                id: encodedId,
                type: 'function',
                function: const oai.FunctionCall(
                  name: 'lookup_capital',
                  arguments: '{"country":"France"}',
                ),
              ),
            ],
          ),
          oai.ChatMessage.tool(
            toolCallId: encodedId,
            content: '{"capital":"Paris"}',
          ),
        ];

        final sanitized = sanitizeMessagesForNonGeminiProvider(messages);

        final assistant = sanitized[1] as oai.AssistantMessage;
        final toolCallId = assistant.toolCalls!.single.id;
        expect(toolCallId, 'call_4');
        expect(toolCallId.length, lessThanOrEqualTo(64));
        // Function payload survives unchanged.
        expect(assistant.toolCalls!.single.function.name, 'lookup_capital');
        expect(
          assistant.toolCalls!.single.function.arguments,
          '{"country":"France"}',
        );

        final toolMsg = sanitized[2] as oai.ToolMessage;
        expect(toolMsg.toolCallId, 'call_4');
        expect(toolMsg.toolCallId.length, lessThanOrEqualTo(64));
        expect(toolMsg.content, '{"capital":"Paris"}');
      },
    );

    test(
      'preserves plain (non-encoded) ids unchanged',
      () {
        final messages = <oai.ChatMessage>[
          oai.ChatMessage.user('hello'),
          const oai.AssistantMessage(
            toolCalls: [
              oai.ToolCall(
                id: 'call_abc',
                type: 'function',
                function: oai.FunctionCall(
                  name: 'do_thing',
                  arguments: '{}',
                ),
              ),
            ],
          ),
          oai.ChatMessage.tool(toolCallId: 'call_abc', content: 'ok'),
        ];

        final sanitized = sanitizeMessagesForNonGeminiProvider(messages);

        // No re-allocation needed when nothing required sanitization.
        expect(identical(sanitized[1], messages[1]), isTrue);
        expect(identical(sanitized[2], messages[2]), isTrue);

        final assistant = sanitized[1] as oai.AssistantMessage;
        expect(assistant.toolCalls!.single.id, 'call_abc');
        final toolMsg = sanitized[2] as oai.ToolMessage;
        expect(toolMsg.toolCallId, 'call_abc');
      },
    );

    test(
      'leaves user / system / developer messages untouched and never mutates input',
      () {
        final signatureBytes = [1, 2, 3, 4];
        final encodedId = encodeThoughtSignatureInToolCallId(
          signatureBase64: base64Encode(signatureBytes),
          originalId: 'call_x',
        );
        final original = <oai.ChatMessage>[
          oai.ChatMessage.system('be helpful'),
          oai.ChatMessage.user('please'),
          oai.AssistantMessage(
            toolCalls: [
              oai.ToolCall(
                id: encodedId,
                type: 'function',
                function: const oai.FunctionCall(name: 'fn', arguments: '{}'),
              ),
            ],
          ),
        ];

        final snapshot = [...original];
        final sanitized = sanitizeMessagesForNonGeminiProvider(original);

        // Input list is not mutated.
        expect(original.length, snapshot.length);
        for (var i = 0; i < original.length; i++) {
          expect(identical(original[i], snapshot[i]), isTrue);
        }

        // System + user pass through by reference.
        expect(identical(sanitized[0], original[0]), isTrue);
        expect(identical(sanitized[1], original[1]), isTrue);

        // Assistant tool_call.id is sanitized.
        final assistant = sanitized[2] as oai.AssistantMessage;
        expect(assistant.toolCalls!.single.id, 'call_x');
      },
    );

    test('mixed batch: some encoded, some plain — only encoded ids change', () {
      final encodedId = encodeThoughtSignatureInToolCallId(
        signatureBase64: base64Encode([9, 8, 7]),
        originalId: 'call_signed',
      );

      final messages = <oai.ChatMessage>[
        oai.AssistantMessage(
          toolCalls: [
            const oai.ToolCall(
              id: 'call_plain',
              type: 'function',
              function: oai.FunctionCall(name: 'fn1', arguments: '{}'),
            ),
            oai.ToolCall(
              id: encodedId,
              type: 'function',
              function: const oai.FunctionCall(name: 'fn2', arguments: '{}'),
            ),
          ],
        ),
      ];

      final sanitized = sanitizeMessagesForNonGeminiProvider(messages);
      final assistant = sanitized.single as oai.AssistantMessage;

      expect(assistant.toolCalls![0].id, 'call_plain');
      expect(assistant.toolCalls![1].id, 'call_signed');
    });
  });
}
