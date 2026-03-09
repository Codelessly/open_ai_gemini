import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:openai_dart/openai_dart.dart' as oai;

import '../../mappers/tool_mapper.dart';
import '../../utils/logger.dart';
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

  const GeminiRequestConversionResult({
    required this.contents,
    this.systemInstruction,
    this.tools,
    this.toolConfig,
    this.generationConfig,
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
  static GeminiRequestConversionResult convert(
    oai.ChatCompletionCreateRequest request, {
    Map<String, String>? thoughtSignatures,
  }) {
    // Convert messages.
    final messageResult = MessageContentConverter.toGemini(
      request.messages,
      thoughtSignatures: thoughtSignatures,
    );

    // Convert tools.
    final geminiTool = ToolMapper.toGeminiTools(request.tools);
    final tools = geminiTool != null ? [geminiTool] : null;

    // Convert tool choice.
    final toolConfig = ToolMapper.toGeminiToolConfig(request.toolChoice);

    // Convert generation config.
    final generationConfig = _buildGenerationConfig(request);

    // Log unsupported parameters.
    _logUnsupported(request);

    return GeminiRequestConversionResult(
      contents: messageResult.contents,
      systemInstruction: messageResult.systemInstruction,
      tools: tools,
      toolConfig: toolConfig,
      generationConfig: generationConfig,
    );
  }

  static gai.GenerationConfig? _buildGenerationConfig(
    oai.ChatCompletionCreateRequest request,
  ) {
    final maxTokens = request.maxCompletionTokens ?? request.maxTokens;

    // Map response format.
    String? responseMimeType;
    Map<String, dynamic>? responseSchema;
    final responseFormat = request.responseFormat;
    if (responseFormat != null) {
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
    }

    // Map reasoning effort to thinking config.
    final thinkingConfig = _buildThinkingConfig(request.reasoningEffort);

    final hasAnyConfig = maxTokens != null ||
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

  static gai.ThinkingConfig? _buildThinkingConfig(
    oai.ReasoningEffort? effort,
  ) {
    if (effort == null) return null;

    final level = switch (effort) {
      oai.ReasoningEffort.low => gai.ThinkingLevel.low,
      oai.ReasoningEffort.medium => gai.ThinkingLevel.medium,
      oai.ReasoningEffort.high => gai.ThinkingLevel.high,
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
