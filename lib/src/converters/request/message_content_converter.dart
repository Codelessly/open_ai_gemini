import 'dart:convert';

import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:openai_dart/openai_dart.dart' as oai;

import '../../mappers/tool_mapper.dart';
import '../../models/media_attachment.dart';
import '../../utils/logger.dart';
import '../../utils/sanitize_unicode.dart';
import '../../utils/thought_signature_utils.dart';

/// Result of converting OpenAI messages to Gemini format.
class MessageConversionResult {
  /// The conversation contents for the Gemini API.
  final List<gai.Content> contents;

  /// The system instruction extracted from system/developer messages.
  final gai.Content? systemInstruction;

  const MessageConversionResult({
    required this.contents,
    this.systemInstruction,
  });
}

/// Converts OpenAI message lists to Gemini content format and vice versa.
class MessageContentConverter {
  const MessageContentConverter._();

  // ---------------------------------------------------------------------------
  // OpenAI → Gemini
  // ---------------------------------------------------------------------------

  /// Converts a list of OpenAI [oai.ChatMessage]s to Gemini [gai.Content]s,
  /// extracting system instructions separately.
  ///
  /// [thoughtSignatures] maps tool call IDs to base64-encoded thought
  /// signatures that must be re-injected into function call parts for
  /// Gemini 3+ models.
  ///
  /// [mediaAttachments] maps message indices (in the original [messages] list)
  /// to media attachments that should be re-injected into the corresponding
  /// Gemini content parts. This is used to round-trip binary data from model
  /// responses through the OpenAI format.
  ///
  /// [modelId] is the target Gemini model ID. Used to determine whether
  /// Gemini 3-specific behaviors are needed (e.g., sentinel thought signatures).
  ///
  /// [sourceProvider] and [sourceModel] identify the provider/model that
  /// generated the assistant messages. When these match the target model,
  /// thinking blocks and thought signatures are preserved as-is. When they
  /// differ, thinking blocks are converted to plain text and signatures are
  /// dropped.
  ///
  /// [normalizeToolCallIds] if true, normalizes tool call IDs by stripping
  /// special characters and capping at 64 characters.
  static MessageConversionResult toGemini(
    List<oai.ChatMessage> messages, {
    Map<String, String>? thoughtSignatures,
    Map<int, List<MediaAttachment>>? mediaAttachments,
    String? modelId,
    String? sourceProvider,
    String? sourceModel,
    bool normalizeToolCallIds = false,
  }) {
    final systemParts = <String>[];
    final contents = <gai.Content>[];

    // Accumulator for consecutive tool messages that must be merged into a
    // single user-role content.
    var pendingToolParts = <gai.Part>[];

    void flushToolParts() {
      if (pendingToolParts.isNotEmpty) {
        contents.add(gai.Content(role: 'user', parts: pendingToolParts));
        pendingToolParts = [];
      }
    }

    for (var i = 0; i < messages.length; i++) {
      final message = messages[i];
      switch (message) {
        case oai.SystemMessage(:final content):
          systemParts.add(sanitizeSurrogates(content));

        case oai.DeveloperMessage(:final content):
          systemParts.add(sanitizeSurrogates(content));

        case oai.UserMessage():
          flushToolParts();
          contents.add(_convertUserMessage(message));

        case oai.AssistantMessage():
          flushToolParts();
          contents.add(
            _convertAssistantMessage(
              message,
              thoughtSignatures: thoughtSignatures,
              attachments: mediaAttachments?[i],
              modelId: modelId,
              sourceProvider: sourceProvider,
              sourceModel: sourceModel,
              normalizeIds: normalizeToolCallIds,
            ),
          );

        case oai.ToolMessage():
          pendingToolParts.add(
            _convertToolMessage(message, messages, modelId: modelId),
          );
      }
    }
    flushToolParts();

    final systemInstruction = systemParts.isNotEmpty
        ? gai.Content(
            parts: [gai.TextPart(systemParts.join('\n\n'))],
          )
        : null;

    return MessageConversionResult(
      contents: contents,
      systemInstruction: systemInstruction,
    );
  }

  static gai.Content _convertUserMessage(oai.UserMessage message) {
    final content = message.content;
    final geminiParts = <gai.Part>[];

    switch (content) {
      case oai.UserTextContent(:final text):
        geminiParts.add(gai.TextPart(sanitizeSurrogates(text)));

      case oai.UserPartsContent(:final parts):
        for (final part in parts) {
          switch (part) {
            case oai.TextContentPart(:final text):
              geminiParts.add(gai.TextPart(sanitizeSurrogates(text)));

            case oai.ImageContentPart(:final url):
              geminiParts.add(_convertImageUrl(url));

            case oai.AudioContentPart(:final data, :final format):
              final mimeType = switch (format) {
                oai.AudioFormat.wav => 'audio/wav',
                oai.AudioFormat.mp3 => 'audio/mp3',
                oai.AudioFormat.flac => 'audio/flac',
                oai.AudioFormat.opus => 'audio/opus',
                oai.AudioFormat.pcm16 => 'audio/pcm',
              };
              geminiParts.add(
                gai.InlineDataPart(gai.Blob(mimeType: mimeType, data: data)),
              );
          }
        }
    }

    if (geminiParts.isEmpty) {
      geminiParts.add(gai.TextPart(''));
    }

    return gai.Content(role: 'user', parts: geminiParts);
  }

  static gai.Part _convertImageUrl(String url) {
    // Data URLs: data:image/png;base64,iVBOR...
    final dataUriMatch = RegExp(
      r'^data:([^;]+);base64,(.+)$',
      dotAll: true,
    ).firstMatch(url);
    if (dataUriMatch != null) {
      return gai.InlineDataPart(
        gai.Blob(
          mimeType: dataUriMatch.group(1)!,
          data: dataUriMatch.group(2)!,
        ),
      );
    }

    // HTTP(S) URLs → FileData.
    if (url.startsWith('http://') || url.startsWith('https://')) {
      return gai.FileDataPart(gai.FileData(fileUri: url));
    }

    // GCS URIs.
    if (url.startsWith('gs://')) {
      return gai.FileDataPart(gai.FileData(fileUri: url));
    }

    // Fallback: treat as base64 image data.
    return gai.InlineDataPart(
      gai.Blob(mimeType: 'image/jpeg', data: url),
    );
  }

  static gai.Content _convertAssistantMessage(
    oai.AssistantMessage message, {
    Map<String, String>? thoughtSignatures,
    List<MediaAttachment>? attachments,
    String? modelId,
    String? sourceProvider,
    String? sourceModel,
    bool normalizeIds = false,
  }) {
    final parts = <gai.Part>[];

    // Determine if the assistant message came from the same provider/model.
    // Only keep thinking blocks as thought:true when same provider AND model.
    // When source info is not available, default to NOT marking as thought
    // (safer fallback that avoids sending stale thinking to wrong model).
    final isSameProviderAndModel =
        sourceProvider != null && sourceModel != null && sourceProvider == 'gemini' && sourceModel == modelId;

    // Add reasoning content.
    final reasoning = message.reasoningContent;
    if (reasoning != null && reasoning.trim().isNotEmpty) {
      if (isSameProviderAndModel) {
        // Same provider/model: keep as thought.
        parts.add(
          gai.TextPart(
            sanitizeSurrogates(reasoning),
            thought: true,
          ),
        );
      } else {
        // Different provider or no source info: convert to plain text.
        parts.add(gai.TextPart(sanitizeSurrogates(reasoning)));
      }
    }

    // Add text content (skip empty/whitespace-only).
    final textContent = message.content;
    if (textContent != null && textContent.trim().isNotEmpty) {
      // Resolve text signature for same-provider messages.
      List<int>? textSignature;
      if (isSameProviderAndModel && thoughtSignatures != null) {
        final sigBase64 = resolveThoughtSignature(
          isSameProviderAndModel: true,
          signature: thoughtSignatures['__last_text__'],
        );
        if (sigBase64 != null) {
          textSignature = base64Decode(sigBase64);
        }
      }

      parts.add(
        gai.TextPart(
          sanitizeSurrogates(textContent),
          thoughtSignature: textSignature,
        ),
      );
    }

    // Re-inject media attachments that were extracted during Gemini → OpenAI
    // conversion.
    if (attachments != null) {
      for (final attachment in attachments) {
        parts.add(_mediaAttachmentToPart(attachment));
      }
    }

    // Add tool calls as function call parts.
    if (message.toolCalls != null) {
      for (final toolCall in message.toolCalls!) {
        Map<String, dynamic>? args;
        try {
          final parsed = jsonDecode(toolCall.function.arguments);
          if (parsed is Map<String, dynamic>) {
            args = parsed;
          } else {
            args = {'value': parsed};
          }
        } catch (_) {
          args = {'value': toolCall.function.arguments};
        }

        // Resolve thought signature for this tool call.
        List<int>? signature;
        if (isSameProviderAndModel && thoughtSignatures != null) {
          final sigBase64 = resolveThoughtSignature(
            isSameProviderAndModel: true,
            signature: thoughtSignatures[toolCall.id],
          );
          if (sigBase64 != null) {
            signature = base64Decode(sigBase64);
          }
        }

        // Gemini 3 requires thoughtSignature on ALL function calls when
        // thinking mode is enabled. Use the sentinel for unsigned calls.
        if (signature == null && isGemini3Model(modelId)) {
          signature = skipThoughtSignatureBytes;
        }

        // Optionally normalize tool call IDs.
        final id = normalizeIds ? normalizeToolCallId(toolCall.id) : null;

        parts.add(
          gai.FunctionCallPart(
            gai.FunctionCall(
              id: id,
              name: toolCall.function.name,
              args: args,
            ),
            thoughtSignature: signature,
          ),
        );
      }
    }

    if (parts.isEmpty) {
      parts.add(gai.TextPart(''));
    }

    return gai.Content(role: 'model', parts: parts);
  }

  /// Converts a [MediaAttachment] to a Gemini [gai.Part].
  static gai.Part _mediaAttachmentToPart(MediaAttachment attachment) {
    if (attachment.isFile) {
      return gai.FileDataPart(
        gai.FileData(
          fileUri: attachment.fileUri!,
          mimeType: attachment.mimeType,
        ),
      );
    }
    return gai.InlineDataPart(
      gai.Blob(mimeType: attachment.mimeType, data: attachment.data!),
    );
  }

  static gai.Part _convertToolMessage(
    oai.ToolMessage message,
    List<oai.ChatMessage> allMessages, {
    String? modelId,
  }) {
    // Resolve the function name from the preceding assistant message's
    // tool calls by matching on toolCallId.
    final functionName = _resolveToolName(message.toolCallId, allMessages);
    return ToolMapper.toGeminiFunctionResponse(
      functionName: functionName,
      content: sanitizeSurrogates(message.content),
      modelId: modelId,
    );
  }

  /// Finds the function name for a given tool call ID by scanning preceding
  /// assistant messages.
  static String _resolveToolName(
    String toolCallId,
    List<oai.ChatMessage> messages,
  ) {
    for (final msg in messages.reversed) {
      if (msg case oai.AssistantMessage(:final toolCalls?)) {
        for (final call in toolCalls) {
          if (call.id == toolCallId) {
            return call.function.name;
          }
        }
      }
    }
    GeminiOpenAILogger.warn(
      'Could not resolve function name for tool call ID "$toolCallId". '
      'Using toolCallId as function name.',
    );
    return toolCallId;
  }

  // ---------------------------------------------------------------------------
  // Gemini → OpenAI
  // ---------------------------------------------------------------------------

  /// Converts a Gemini [gai.Content] to an OpenAI [oai.ChatMessage].
  ///
  /// Returns a [GeminiMessageConversionResult] containing the OpenAI message
  /// and any extracted thought signatures or media attachments.
  static GeminiMessageConversionResult toOpenAI(
    gai.Content content, {
    String Function()? generateToolCallId,
  }) {
    final role = content.role;

    if (role == 'user') {
      return _convertGeminiUserContent(content);
    } else {
      return _convertGeminiModelContent(
        content,
        generateToolCallId: generateToolCallId,
      );
    }
  }

  static GeminiMessageConversionResult _convertGeminiUserContent(
    gai.Content content,
  ) {
    final textParts = <String>[];
    final contentParts = <oai.ContentPart>[];
    final functionResponses = <_FunctionResponseInfo>[];
    var hasMediaParts = false;

    for (final part in content.parts) {
      switch (part) {
        case gai.TextPart(:final text):
          textParts.add(text);
          contentParts.add(oai.ContentPart.text(text));

        case gai.FunctionResponsePart(:final functionResponse):
          functionResponses.add(
            _FunctionResponseInfo(
              name: functionResponse.name,
              response: functionResponse.response,
            ),
          );

        case gai.InlineDataPart(:final inlineData):
          hasMediaParts = true;
          final mimeType = inlineData.mimeType;
          final data = inlineData.data;

          if (mimeType.startsWith('audio/')) {
            final format = _mimeToAudioFormat(mimeType);
            if (format != null) {
              contentParts.add(
                oai.ContentPart.inputAudio(data: data, format: format),
              );
            } else {
              // Unsupported audio format — encode as image data URL fallback.
              contentParts.add(
                oai.ContentPart.imageUrl('data:$mimeType;base64,$data'),
              );
            }
          } else {
            // Images, PDFs, videos, etc. → data URL.
            contentParts.add(
              oai.ContentPart.imageUrl('data:$mimeType;base64,$data'),
            );
          }

        case gai.FileDataPart(:final fileData):
          hasMediaParts = true;
          contentParts.add(oai.ContentPart.imageUrl(fileData.fileUri));

        default:
          break;
      }
    }

    // If this user content contains function responses, convert to tool
    // messages.
    if (functionResponses.isNotEmpty) {
      final messages = <oai.ChatMessage>[];
      for (final resp in functionResponses) {
        messages.add(
          oai.ChatMessage.tool(
            toolCallId: resp.name,
            content: jsonEncode(resp.response),
          ),
        );
      }
      return GeminiMessageConversionResult(messages: messages);
    }

    // If there are media parts, use multipart content.
    if (hasMediaParts) {
      return GeminiMessageConversionResult(
        messages: [oai.ChatMessage.user(contentParts)],
      );
    }

    return GeminiMessageConversionResult(
      messages: [oai.ChatMessage.user(textParts.join('\n'))],
    );
  }

  static GeminiMessageConversionResult _convertGeminiModelContent(
    gai.Content content, {
    String Function()? generateToolCallId,
  }) {
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

    final textContent = textParts.isNotEmpty ? textParts.join('') : null;
    final reasoning = reasoningParts.isNotEmpty ? reasoningParts.join('') : null;

    return GeminiMessageConversionResult(
      messages: [
        oai.AssistantMessage(
          content: textContent,
          toolCalls: toolCalls.isNotEmpty ? toolCalls : null,
          reasoningContent: reasoning,
        ),
      ],
      thoughtSignatures: thoughtSignatures.isNotEmpty ? thoughtSignatures : null,
      mediaAttachments: media.isNotEmpty ? media : null,
    );
  }

  /// Maps a MIME type string to an OpenAI [oai.AudioFormat], or `null` if
  /// the format is not supported.
  static oai.AudioFormat? _mimeToAudioFormat(String mimeType) {
    return switch (mimeType) {
      'audio/wav' || 'audio/x-wav' => oai.AudioFormat.wav,
      'audio/mp3' || 'audio/mpeg' => oai.AudioFormat.mp3,
      'audio/flac' || 'audio/x-flac' => oai.AudioFormat.flac,
      'audio/opus' || 'audio/ogg' => oai.AudioFormat.opus,
      'audio/pcm' || 'audio/L16' => oai.AudioFormat.pcm16,
      _ => null,
    };
  }
}

/// Result of converting a Gemini message to OpenAI format.
class GeminiMessageConversionResult {
  /// The converted OpenAI message(s).
  ///
  /// A single Gemini content with multiple function responses may produce
  /// multiple tool messages.
  final List<oai.ChatMessage> messages;

  /// Thought signatures extracted from function call parts, keyed by
  /// tool call ID. Values are base64-encoded.
  final Map<String, String>? thoughtSignatures;

  /// Media attachments extracted from model responses.
  ///
  /// Since OpenAI's `AssistantMessage` only supports text content, any binary
  /// data (images, audio, files) from Gemini model responses are captured here.
  /// Pass these back via [MessageContentConverter.toGemini]'s
  /// `mediaAttachments` parameter to round-trip them.
  final List<MediaAttachment>? mediaAttachments;

  const GeminiMessageConversionResult({
    required this.messages,
    this.thoughtSignatures,
    this.mediaAttachments,
  });
}

class _FunctionResponseInfo {
  final String name;
  final Map<String, dynamic> response;

  const _FunctionResponseInfo({required this.name, required this.response});
}
