import 'dart:convert';

import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:openai_dart/openai_dart.dart' as oai;

import '../../mappers/finish_reason_mapper.dart';
import '../../models/media_attachment.dart';

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
          final id = generateToolCallId?.call() ?? 'call_${toolCallIndex}_${functionCall.name}';
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
          if (thoughtSignature != null && thoughtSignature.isNotEmpty) {
            thoughtSignatures[id] = base64Encode(thoughtSignature);
          }
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
