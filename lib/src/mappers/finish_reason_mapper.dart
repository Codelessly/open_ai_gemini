import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:openai_dart/openai_dart.dart' as oai;

/// Maps Gemini finish reasons to OpenAI finish reasons and vice versa.
class FinishReasonMapper {
  const FinishReasonMapper._();

  /// Converts a Gemini [gai.FinishReason] to an OpenAI [oai.FinishReason].
  ///
  /// If tool calls are present, returns [oai.FinishReason.toolCalls] regardless
  /// of the Gemini finish reason (matching OpenAI behavior).
  static oai.FinishReason? toOpenAI(
    gai.FinishReason? reason, {
    bool hasToolCalls = false,
  }) {
    if (hasToolCalls) return oai.FinishReason.toolCalls;

    return switch (reason) {
      gai.FinishReason.stop => oai.FinishReason.stop,
      gai.FinishReason.maxTokens => oai.FinishReason.length,
      gai.FinishReason.safety => oai.FinishReason.contentFilter,
      gai.FinishReason.recitation => oai.FinishReason.contentFilter,
      gai.FinishReason.blocklist => oai.FinishReason.contentFilter,
      gai.FinishReason.prohibitedContent => oai.FinishReason.contentFilter,
      gai.FinishReason.spii => oai.FinishReason.contentFilter,
      gai.FinishReason.malformedFunctionCall => oai.FinishReason.stop,
      gai.FinishReason.other => oai.FinishReason.stop,
      gai.FinishReason.unspecified => oai.FinishReason.stop,
      null => null,
    };
  }
}
