import 'dart:convert';

import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:openai_dart/openai_dart.dart' as oai;

import '../../mappers/finish_reason_mapper.dart';
import '../../models/media_attachment.dart';
import '../../utils/thought_signature_utils.dart';

/// Result of converting a Gemini response to OpenAI format.
class GeminiResponseConversionResult {
  /// The OpenAI-formatted chat completion.
  final oai.ChatCompletion completion;

  /// Thought signatures extracted from function call parts, keyed by
  /// tool call ID. Values are base64-encoded.
  ///
  /// These must be preserved and passed back when converting the next
  /// request to Gemini format for Gemini 3+ models.
  final Map<String, String> thoughtSignatures;

  /// Media attachments extracted from the response.
  ///
  /// Since OpenAI's `AssistantMessage` only supports text content, any binary
  /// data (images, audio, files) from Gemini model responses are captured here.
  /// Store these alongside the message and pass them back via
  /// [ChatCompletionRequestConverter.convert]'s `mediaAttachments` parameter
  /// to round-trip them.
  final List<MediaAttachment> mediaAttachments;

  const GeminiResponseConversionResult({
    required this.completion,
    this.thoughtSignatures = const {},
    this.mediaAttachments = const [],
  });
}

/// Converts a Gemini [gai.GenerateContentResponse] to an OpenAI
/// [oai.ChatCompletion].
class ChatCompletionResponseConverter {
  const ChatCompletionResponseConverter._();

  /// Converts a Gemini response to an OpenAI chat completion.
  ///
  /// [model] is the model name to include in the response.
  /// [generateToolCallId] is an optional function to generate unique IDs for
  /// tool calls. Defaults to `call_{index}_{name}`.
  static GeminiResponseConversionResult convert(
    gai.GenerateContentResponse response, {
    required String model,
    String Function()? generateToolCallId,
  }) {
    final candidate = response.candidates?.firstOrNull;
    final content = candidate?.content;

    if (content == null) {
      return GeminiResponseConversionResult(
        completion: oai.ChatCompletion(
          id: response.responseId ?? _generateId(),
          choices: [
            oai.ChatChoice(
              index: 0,
              message: const oai.AssistantMessage(content: ''),
              finishReason: FinishReasonMapper.toOpenAI(
                candidate?.finishReason,
              ),
            ),
          ],
          created: DateTime.now().millisecondsSinceEpoch ~/ 1000,
          model: model,
          object: 'chat.completion',
          usage: _convertUsage(response.usageMetadata),
          provider: 'gemini',
        ),
      );
    }

    final textParts = <String>[];
    final reasoningParts = <String>[];
    final toolCalls = <oai.ToolCall>[];
    final thoughtSignatures = <String, String>{};
    final media = <MediaAttachment>[];
    var toolCallIndex = 0;

    for (final part in content.parts) {
      switch (part) {
        case gai.TextPart(:final text, :final thought, :final thoughtSignature):
          if (thought == true) {
            reasoningParts.add(text);
          } else {
            textParts.add(text);
          }
          if (thoughtSignature != null && thoughtSignature.isNotEmpty) {
            thoughtSignatures['__last_text__'] = base64Encode(thoughtSignature);
          }

        case gai.FunctionCallPart(:final functionCall, :final thoughtSignature):
          final rawId = generateToolCallId?.call() ?? 'call_${toolCallIndex}_${functionCall.name}';

          // Encode the thought signature directly into the tool_call.id when
          // present. This makes the signature survive
          // `assistantMessage.toJson() → JSON store → fromJson()` round-trips
          // through callers (e.g. agent_kit's history rehydration) that do
          // not know about thought signatures — the OpenAI ChatMessage schema
          // has no field for the signature, but it does carry tool_call.id
          // verbatim. The encoded id is opaque to the model; Gemini receives
          // only the function `name` + `args` on the outgoing path.
          final String id;
          if (thoughtSignature != null && thoughtSignature.isNotEmpty) {
            final sigBase64 = base64Encode(thoughtSignature);
            id = encodeThoughtSignatureInToolCallId(
              signatureBase64: sigBase64,
              originalId: rawId,
            );
            // Mirror into the legacy in-memory map keyed by the encoded id so
            // any caller reading `thoughtSignatures` directly still finds it.
            thoughtSignatures[id] = sigBase64;
          } else {
            id = rawId;
          }

          toolCalls.add(
            oai.ToolCall(
              id: id,
              type: 'function',
              function: oai.FunctionCall(
                name: functionCall.name,
                arguments: jsonEncode(functionCall.args ?? {}),
              ),
            ),
          );
          toolCallIndex++;

        case gai.ThoughtSignaturePart(:final thoughtSignature):
          if (thoughtSignature.isNotEmpty) {
            thoughtSignatures['__last_text__'] = base64Encode(thoughtSignature);
          }

        case gai.InlineDataPart(:final inlineData):
          media.add(
            MediaAttachment.inline(
              mimeType: inlineData.mimeType,
              data: inlineData.data,
            ),
          );

        case gai.FileDataPart(:final fileData):
          media.add(
            MediaAttachment.file(
              mimeType: fileData.mimeType ?? 'application/octet-stream',
              fileUri: fileData.fileUri,
            ),
          );

        default:
          break;
      }
    }

    final hasToolCalls = toolCalls.isNotEmpty;
    final textContent = textParts.isNotEmpty ? textParts.join('') : null;
    final reasoning = reasoningParts.isNotEmpty ? reasoningParts.join('') : null;

    final finishReason = FinishReasonMapper.toOpenAI(
      candidate?.finishReason,
      hasToolCalls: hasToolCalls,
    );

    final completion = oai.ChatCompletion(
      id: response.responseId ?? _generateId(),
      choices: [
        oai.ChatChoice(
          index: 0,
          message: oai.AssistantMessage(
            content: textContent,
            toolCalls: hasToolCalls ? toolCalls : null,
            reasoningContent: reasoning,
          ),
          finishReason: finishReason,
        ),
      ],
      created: DateTime.now().millisecondsSinceEpoch ~/ 1000,
      model: model,
      object: 'chat.completion',
      usage: _convertUsage(response.usageMetadata),
      provider: 'gemini',
    );

    return GeminiResponseConversionResult(
      completion: completion,
      thoughtSignatures: thoughtSignatures,
      mediaAttachments: media,
    );
  }

  /// Maps Gemini's [UsageMetadata] to the OpenAI-shaped [oai.Usage].
  ///
  /// `cachedContentTokenCount` (the count of prompt tokens served from
  /// Google's implicit context cache OR an explicit `cachedContents/…`
  /// reference) lands in `promptTokensDetails.cachedTokens` — the same
  /// slot the OpenAI SDK uses for its own cache reads. Callers that read
  /// `usage.promptTokensDetails.cachedTokens` get a uniform metric across
  /// providers.
  ///
  /// ### Implicit-caching notes (Google-side behavior)
  ///
  /// Google's documentation lists per-family minimums (4096 tokens for
  /// Gemini 3 / 3.1, 2048 for 2.5 Flash / Pro) but in practice the
  /// behavior is **model-specific** and not always reliable near those
  /// minimums:
  ///
  ///   * `gemini-3-flash-preview`  — caches reliably at any prefix ≥ ~4K
  ///   * `gemini-3.1-pro-preview`  — caches reliably at any prefix ≥ ~4K
  ///   * `gemini-3.5-flash`        — INCONSISTENT in the ~7-15K range;
  ///                                  caches at ~6K and ≥ ~20K but flips
  ///                                  on/off in between (probed in
  ///                                  `open_ai_gemini/tool/probe_3_5_flash_cache.dart`)
  ///
  /// If you need reliable caching at a specific prefix size on a model
  /// whose implicit-cache behavior is flaky, use the explicit
  /// `cachedContents` API plumbed through [GeminiOpenAIClient.cachedContent].
  /// That bypasses implicit caching and lets you control the cache name
  /// and TTL directly.
  static oai.Usage? _convertUsage(gai.UsageMetadata? metadata) {
    if (metadata == null) return null;

    final prompt = metadata.promptTokenCount ?? 0;
    final completion = metadata.candidatesTokenCount ?? 0;
    final cached = metadata.cachedContentTokenCount;

    return oai.Usage(
      promptTokens: prompt,
      completionTokens: completion,
      totalTokens: metadata.totalTokenCount ?? (prompt + completion),
      promptTokensDetails: cached != null ? oai.PromptTokensDetails(cachedTokens: cached) : null,
    );
  }

  static String _generateId() {
    return 'chatcmpl-gemini-${DateTime.now().millisecondsSinceEpoch}';
  }
}
