import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:openai_dart/openai_dart.dart' as oai;

import '../../mappers/tool_mapper.dart';
import '../../utils/logger.dart';
import '../../utils/thought_signature_utils.dart';
import 'message_content_converter.dart';

/// Result of converting an OpenAI request to Gemini format.
class GeminiRequestConversionResult {
  /// The conversation contents for the Gemini API.
  final List<gai.Content> contents;

  /// The system instruction extracted from system/developer messages.
  final gai.Content? systemInstruction;

  /// The Gemini tools (function declarations).
  final List<gai.Tool>? tools;

  /// The tool configuration (function calling mode).
  final gai.ToolConfig? toolConfig;

  /// The generation configuration.
  final gai.GenerationConfig? generationConfig;

  /// The resource name of cached content to use (e.g., `cachedContents/abc`).
  final String? cachedContent;

  const GeminiRequestConversionResult({
    required this.contents,
    this.systemInstruction,
    this.tools,
    this.toolConfig,
    this.generationConfig,
    this.cachedContent,
  });
}

/// Converts an OpenAI [oai.ChatCompletionCreateRequest] to Gemini API
/// request components.
class ChatCompletionRequestConverter {
  const ChatCompletionRequestConverter._();

  /// Converts an OpenAI chat completion request to Gemini request components.
  ///
  /// [thoughtSignatures] maps tool call IDs to base64-encoded thought
  /// signatures that must be preserved for Gemini 3+ models.
  ///
  /// [sourceProvider] and [sourceModel] identify the provider/model that
  /// generated the assistant messages. Used for cross-provider awareness.
  static GeminiRequestConversionResult convert(
    oai.ChatCompletionCreateRequest request, {
    Map<String, String>? thoughtSignatures,
    String? cachedContent,
    String? sourceProvider,
    String? sourceModel,
  }) {
    // Convert messages. Pass the request's model as the target modelId.
    final messageResult = MessageContentConverter.toGemini(
      request.messages,
      thoughtSignatures: thoughtSignatures,
      modelId: request.model,
      sourceProvider: sourceProvider,
      sourceModel: sourceModel,
    );

    // Convert tools.
    final tools = buildTools(request);

    // Convert tool choice.
    final toolConfig = buildToolConfig(request);

    // Convert generation config.
    final generationConfig = buildGenerationConfig(request);

    // Log unsupported parameters.
    _logUnsupported(request);

    return GeminiRequestConversionResult(
      contents: messageResult.contents,
      systemInstruction: messageResult.systemInstruction,
      tools: tools,
      toolConfig: toolConfig,
      generationConfig: generationConfig,
      cachedContent: cachedContent,
    );
  }

  /// Builds the list of Gemini tools from an OpenAI request.
  static List<gai.Tool>? buildTools(oai.ChatCompletionCreateRequest request) {
    final geminiTool = ToolMapper.toGeminiTools(request.tools);
    return geminiTool != null ? [geminiTool] : null;
  }

  /// Builds the Gemini tool configuration from an OpenAI request's tool choice.
  static gai.ToolConfig? buildToolConfig(
    oai.ChatCompletionCreateRequest request,
  ) {
    return ToolMapper.toGeminiToolConfig(request.toolChoice);
  }

  /// Builds the Gemini generation configuration from an OpenAI request.
  static gai.GenerationConfig? buildGenerationConfig(
    oai.ChatCompletionCreateRequest request,
  ) {
    final maxTokens = request.maxCompletionTokens ?? request.maxTokens;

    // Gemini's function calling is incompatible with `responseSchema` /
    // `responseMimeType = application/json`. Per Google's documented
    // behaviour, when both tools and a JSON response schema are provided,
    // the schema is silently ignored AND the model frequently emits a
    // spurious function call instead of the requested JSON. Empirically
    // (Gemini 3 Flash Preview, 2026-05) the response in this combo is
    // non-deterministic — sometimes a tool call, sometimes a truncated
    // text JSON, sometimes a complete text JSON — which surfaces as
    // intermittent "Failed to parse JSON response" errors upstream.
    //
    // OpenAI's spec does allow `tools + response_format` to coexist; this
    // converter therefore reconciles by dropping the schema (not the
    // tools) when both are present. Callers that need strict structured
    // output alongside tools must enforce the schema via a forced tool
    // call themselves, not via `response_format`.
    final hasTools = request.tools != null && request.tools!.isNotEmpty;

    // Map response format.
    String? responseMimeType;
    Map<String, dynamic>? responseSchema;
    final responseFormat = request.responseFormat;
    if (responseFormat != null && !hasTools) {
      switch (responseFormat) {
        case oai.JsonObjectResponseFormat():
          responseMimeType = 'application/json';
        case oai.JsonSchemaResponseFormat(
          :final name,
          :final schema,
          :final description,
        ):
          responseMimeType = 'application/json';
          responseSchema = ToolMapper.sanitizeSchema({
            'type': 'OBJECT',
            // ignore: use_null_aware_elements
            if (description != null) 'description': description,
            ...schema,
          });
          // Include the schema name as a description prefix if not already
          // present.
          // ignore: unnecessary_non_null_assertion
          if (responseSchema!['description'] == null) {
            responseSchema['description'] = name;
          }
        case oai.TextResponseFormat():
          responseMimeType = 'text/plain';
      }
    } else if (responseFormat != null && hasTools) {
      GeminiOpenAILogger.warn(
        'response_format is incompatible with tools on Gemini and has '
        'been dropped. Gemini ignores responseSchema when tools are '
        'present and frequently emits spurious tool calls instead of '
        'JSON. The schema must be enforced via prompt or a forced tool '
        'call by the caller.',
      );
    }

    // Map reasoning effort to thinking config.
    //
    // For Gemini 3+ models that are being given tools, default to
    // thinking-enabled (low level) when the caller didn't specify a
    // reasoning effort. Without thinking, Gemini 3 does NOT emit
    // `thoughtSignature` on function-call parts — and on the next
    // round-trip (replaying the assistant's function_call alongside
    // the tool result) the API rejects the request with HTTP 400:
    //   "Function call is missing a thought_signature in functionCall
    //    parts. This is required for tools to work correctly."
    // The sentinel fallback this package inserts on unsigned calls
    // (`skip_thought_signature_validator`) is rejected by the API as
    // well, so the only real fix is to make Gemini emit real
    // signatures by enabling thinking.
    final isGemini3 = isGemini3Model(request.model);
    final effectiveEffort = request.reasoningEffort ?? ((hasTools && isGemini3) ? oai.ReasoningEffort.low : null);
    final thinkingConfig = buildThinkingConfig(effectiveEffort);

    final hasAnyConfig =
        maxTokens != null ||
        request.temperature != null ||
        request.topP != null ||
        request.stop != null ||
        responseMimeType != null ||
        responseSchema != null ||
        thinkingConfig != null ||
        request.seed != null;

    if (!hasAnyConfig) return null;

    return gai.GenerationConfig(
      maxOutputTokens: maxTokens,
      temperature: request.temperature,
      topP: request.topP,
      stopSequences: request.stop,
      responseMimeType: responseMimeType,
      responseSchema: responseSchema,
      thinkingConfig: thinkingConfig,
      seed: request.seed,
    );
  }

  /// Builds the Gemini thinking configuration from an OpenAI reasoning effort.
  static gai.ThinkingConfig? buildThinkingConfig(
    oai.ReasoningEffort? effort,
  ) {
    if (effort == null) return null;

    final level = switch (effort) {
      oai.ReasoningEffort.none => gai.ThinkingLevel.low,
      oai.ReasoningEffort.minimal => gai.ThinkingLevel.low,
      oai.ReasoningEffort.low => gai.ThinkingLevel.low,
      oai.ReasoningEffort.medium => gai.ThinkingLevel.medium,
      oai.ReasoningEffort.high => gai.ThinkingLevel.high,
      oai.ReasoningEffort.xhigh => gai.ThinkingLevel.high,
      oai.ReasoningEffort.unknown => gai.ThinkingLevel.medium,
    };

    return gai.ThinkingConfig(
      includeThoughts: true,
      thinkingLevel: level,
    );
  }

  static void _logUnsupported(oai.ChatCompletionCreateRequest request) {
    GeminiOpenAILogger.logUnsupportedParam(
      'frequency_penalty',
      request.frequencyPenalty,
    );
    GeminiOpenAILogger.logUnsupportedParam(
      'presence_penalty',
      request.presencePenalty,
    );
    GeminiOpenAILogger.logUnsupportedParam(
      'logit_bias',
      request.logitBias,
    );
    GeminiOpenAILogger.logUnsupportedParam('logprobs', request.logprobs);
    GeminiOpenAILogger.logUnsupportedParam(
      'top_logprobs',
      request.topLogprobs,
    );
    if (request.n != null && request.n! > 1) {
      GeminiOpenAILogger.warn(
        'Gemini supports only 1 candidate via this translation layer. '
        'n=${request.n} will be ignored.',
      );
    }
  }
}
