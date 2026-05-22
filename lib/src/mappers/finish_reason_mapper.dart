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
      // Content / policy blocks → OpenAI's content_filter category.
      gai.FinishReason.safety => oai.FinishReason.contentFilter,
      gai.FinishReason.recitation => oai.FinishReason.contentFilter,
      gai.FinishReason.blocklist => oai.FinishReason.contentFilter,
      gai.FinishReason.prohibitedContent => oai.FinishReason.contentFilter,
      gai.FinishReason.spii => oai.FinishReason.contentFilter,
      // googleai_dart 6.x added language + image-related block categories
      // and tool-call control reasons. Map block categories to content_filter
      // and the rest to stop (closest OpenAI semantic for "we stopped").
      gai.FinishReason.language => oai.FinishReason.contentFilter,
      gai.FinishReason.imageSafety => oai.FinishReason.contentFilter,
      gai.FinishReason.imageProhibitedContent => oai.FinishReason.contentFilter,
      gai.FinishReason.imageOther => oai.FinishReason.contentFilter,
      gai.FinishReason.imageRecitation => oai.FinishReason.contentFilter,
      gai.FinishReason.noImage => oai.FinishReason.stop,
      gai.FinishReason.unexpectedToolCall => oai.FinishReason.stop,
      gai.FinishReason.tooManyToolCalls => oai.FinishReason.stop,
      gai.FinishReason.missingThoughtSignature => oai.FinishReason.stop,
      gai.FinishReason.malformedResponse => oai.FinishReason.stop,
      gai.FinishReason.malformedFunctionCall => oai.FinishReason.stop,
      gai.FinishReason.other => oai.FinishReason.stop,
      gai.FinishReason.unspecified => oai.FinishReason.stop,
      null => null,
    };
  }
}
