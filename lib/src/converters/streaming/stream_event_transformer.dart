import 'dart:async';
import 'dart:convert';

import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:openai_dart/openai_dart.dart' as oai;

import '../../mappers/finish_reason_mapper.dart';
import '../../models/media_attachment.dart';

/// Transforms a stream of Gemini [gai.GenerateContentResponse] chunks into
/// OpenAI-compatible [oai.ChatStreamEvent]s.
class GeminiStreamEventTransformer
    extends StreamTransformerBase<gai.GenerateContentResponse, oai.ChatStreamEvent> {
  final String _model;
  final String Function()? _generateToolCallId;

  /// Creates a transformer that converts Gemini streaming chunks to OpenAI
  /// streaming events.
  ///
  /// [model] is the model name to include in each chunk.
  /// [generateToolCallId] optionally provides unique IDs for tool calls.
  const GeminiStreamEventTransformer({
    required String model,
    String Function()? generateToolCallId,
  })  : _model = model,
        _generateToolCallId = generateToolCallId;

  @override
  Stream<oai.ChatStreamEvent> bind(
    Stream<gai.GenerateContentResponse> stream,
  ) {
    return _TransformingStream(
      source: stream,
      model: _model,
      generateToolCallId: _generateToolCallId,
    );
  }
}

/// Result of a streaming conversion, containing both the stream of events
/// and any thought signatures extracted during streaming.
class GeminiStreamConversionResult {
  /// The stream of OpenAI-compatible chat stream events.
  final Stream<oai.ChatStreamEvent> events;

  /// A future that completes with thought signatures extracted during
  /// streaming, keyed by tool call ID with base64-encoded values.
  final Future<Map<String, String>> thoughtSignatures;

  /// A future that completes with media attachments extracted during
  /// streaming.
  final Future<List<MediaAttachment>> mediaAttachments;

  const GeminiStreamConversionResult({
    required this.events,
    required this.thoughtSignatures,
    required this.mediaAttachments,
  });
}

/// Converts a Gemini streaming response to OpenAI format, capturing thought
/// signatures.
///
/// This is a convenience function that wraps [GeminiStreamEventTransformer]
/// and also collects thought signatures from the stream.
GeminiStreamConversionResult convertGeminiStream(
  Stream<gai.GenerateContentResponse> geminiStream, {
  required String model,
  String Function()? generateToolCallId,
}) {
  final signaturesCompleter = Completer<Map<String, String>>();
  final mediaCompleter = Completer<List<MediaAttachment>>();
  final signatures = <String, String>{};
  final media = <MediaAttachment>[];
  var toolCallIndex = 0;

  final controller = StreamController<oai.ChatStreamEvent>();
  final state = _StreamState(model: model);
  var isFirstChunk = true;

  geminiStream.listen(
    (chunk) {
      final candidate = chunk.candidates?.firstOrNull;
      final parts = candidate?.content?.parts ?? [];

      // Emit initial chunk with role on first event.
      if (isFirstChunk) {
        controller.add(_buildEvent(
          state: state,
          delta: const oai.ChatDelta(role: 'assistant'),
        ));
        isFirstChunk = false;
      }

      for (final part in parts) {
        switch (part) {
          case gai.TextPart(:final text, :final thought, :final thoughtSignature):
            if (thought == true) {
              controller.add(_buildEvent(
                state: state,
                delta: oai.ChatDelta(reasoningContent: text),
              ));
            } else {
              controller.add(_buildEvent(
                state: state,
                delta: oai.ChatDelta(content: text),
              ));
            }
            if (thoughtSignature != null && thoughtSignature.isNotEmpty) {
              signatures['__last_text__'] = base64Encode(thoughtSignature);
            }

          case gai.FunctionCallPart(:final functionCall, :final thoughtSignature):
            final id = generateToolCallId?.call() ??
                'call_${toolCallIndex}_${functionCall.name}';

            // Emit tool call start delta.
            controller.add(_buildEvent(
              state: state,
              delta: oai.ChatDelta(
                toolCalls: [
                  oai.ToolCallDelta(
                    index: toolCallIndex,
                    id: id,
                    type: 'function',
                    function: oai.FunctionCallDelta(
                      name: functionCall.name,
                      arguments: jsonEncode(functionCall.args ?? {}),
                    ),
                  ),
                ],
              ),
            ));

            if (thoughtSignature != null && thoughtSignature.isNotEmpty) {
              signatures[id] = base64Encode(thoughtSignature);
            }
            toolCallIndex++;

          case gai.ThoughtSignaturePart(:final thoughtSignature):
            if (thoughtSignature.isNotEmpty) {
              signatures['__last_text__'] = base64Encode(thoughtSignature);
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

      // Emit finish reason if present.
      if (candidate?.finishReason != null) {
        final hasToolCalls = toolCallIndex > 0;
        final finishReason = FinishReasonMapper.toOpenAI(
          candidate!.finishReason,
          hasToolCalls: hasToolCalls,
        );
        final usage = chunk.usageMetadata != null
            ? _convertUsage(chunk.usageMetadata!)
            : null;

        controller.add(_buildEvent(
          state: state,
          delta: const oai.ChatDelta(),
          finishReason: finishReason,
          usage: usage,
        ));
      }
    },
    onError: (Object error, StackTrace stackTrace) {
      controller.addError(error, stackTrace);
    },
    onDone: () {
      signaturesCompleter.complete(signatures);
      mediaCompleter.complete(media);
      controller.close();
    },
  );

  return GeminiStreamConversionResult(
    events: controller.stream,
    thoughtSignatures: signaturesCompleter.future,
    mediaAttachments: mediaCompleter.future,
  );
}

class _StreamState {
  final String messageId;
  final String model;
  final int created;

  _StreamState({required this.model})
      : messageId = 'chatcmpl-gemini-${DateTime.now().millisecondsSinceEpoch}',
        created = DateTime.now().millisecondsSinceEpoch ~/ 1000;
}

oai.ChatStreamEvent _buildEvent({
  required _StreamState state,
  required oai.ChatDelta delta,
  oai.FinishReason? finishReason,
  oai.Usage? usage,
}) {
  return oai.ChatStreamEvent(
    id: state.messageId,
    choices: [
      oai.ChatStreamChoice(
        index: 0,
        delta: delta,
        finishReason: finishReason,
      ),
    ],
    created: state.created,
    model: state.model,
    object: 'chat.completion.chunk',
    usage: usage,
    provider: 'gemini',
  );
}

oai.Usage _convertUsage(gai.UsageMetadata metadata) {
  final prompt = metadata.promptTokenCount ?? 0;
  final completion = metadata.candidatesTokenCount ?? 0;
  final cached = metadata.cachedContentTokenCount;

  return oai.Usage(
    promptTokens: prompt,
    completionTokens: completion,
    totalTokens: metadata.totalTokenCount ?? (prompt + completion),
    promptTokensDetails: cached != null
        ? oai.PromptTokensDetails(cachedTokens: cached)
        : null,
  );
}

/// Internal stream implementation for the transformer.
class _TransformingStream extends Stream<oai.ChatStreamEvent> {
  final Stream<gai.GenerateContentResponse> source;
  final String model;
  final String Function()? generateToolCallId;

  _TransformingStream({
    required this.source,
    required this.model,
    this.generateToolCallId,
  });

  @override
  StreamSubscription<oai.ChatStreamEvent> listen(
    void Function(oai.ChatStreamEvent event)? onData, {
    Function? onError,
    void Function()? onDone,
    bool? cancelOnError,
  }) {
    final result = convertGeminiStream(
      source,
      model: model,
      generateToolCallId: generateToolCallId,
    );
    return result.events.listen(
      onData,
      onError: onError,
      onDone: onDone,
      cancelOnError: cancelOnError,
    );
  }
}
