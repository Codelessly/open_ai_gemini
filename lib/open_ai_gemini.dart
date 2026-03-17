/// A translation layer between OpenAI and Google Gemini API specifications.
///
/// Provides bidirectional conversion between OpenAI's chat completion format
/// and Gemini's native API format, enabling storage of all agent conversations
/// in a single OpenAI-compliant data structure.
///
/// ## Key Features
///
/// - **Request conversion**: OpenAI `ChatCompletionCreateRequest` → Gemini
///   `GenerateContentRequest` components
/// - **Response conversion**: Gemini `GenerateContentResponse` → OpenAI
///   `ChatCompletion`
/// - **Streaming**: Gemini streaming chunks → OpenAI `ChatStreamEvent`s
/// - **Tool calling**: Bidirectional tool definition, tool call, and tool
///   result conversion with schema sanitization
/// - **Thought signatures**: Preserves Gemini 3+ cryptographic thought
///   signatures across round-trips
/// - **Thinking/reasoning**: Maps OpenAI `reasoning_effort` to Gemini
///   `thinkingConfig` and surfaces thinking tokens as `reasoningContent`
library;

// Client
export 'src/client/client.dart' show GeminiOpenAIClient;

// Converters
export 'src/converters/request/chat_completion_request_converter.dart'
    show ChatCompletionRequestConverter, GeminiRequestConversionResult;
export 'src/converters/request/message_content_converter.dart'
    show MessageContentConverter, MessageConversionResult, GeminiMessageConversionResult;
export 'src/converters/response/chat_completion_response_converter.dart'
    show ChatCompletionResponseConverter, GeminiResponseConversionResult;
export 'src/converters/streaming/stream_event_transformer.dart'
    show GeminiStreamEventTransformer, GeminiStreamConversionResult, convertGeminiStream;

// Mappers
export 'src/mappers/finish_reason_mapper.dart' show FinishReasonMapper;
export 'src/mappers/tool_mapper.dart' show ToolMapper;

// Models
export 'src/models/media_attachment.dart' show MediaAttachment;

// Utils
export 'src/utils/sanitize_unicode.dart' show sanitizeSurrogates;
export 'src/utils/thought_signature_utils.dart'
    show isValidThoughtSignature, resolveThoughtSignature, isGemini3Model, normalizeToolCallId;
