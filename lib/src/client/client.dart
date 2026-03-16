import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:http/http.dart' as http;
import 'package:openai_dart/openai_dart.dart';
// ignore: implementation_imports, depend_on_referenced_packages
import 'package:googleai_dart/src/resources/cached_contents_resource.dart';
// ignore: implementation_imports, depend_on_referenced_packages
import 'package:openai_dart/src/client/request_builder.dart';

import '../converters/request/chat_completion_request_converter.dart';
import '../converters/request/message_content_converter.dart';
import '../converters/response/chat_completion_response_converter.dart';
import '../converters/streaming/stream_event_transformer.dart';

/// A client that exposes OpenAI's API interface but uses Google's Gemini models.
///
/// This client extends [OpenAIClient] and can be used as a drop-in replacement
/// anywhere an [OpenAIClient] is expected. It translates OpenAI API calls to
/// Gemini's native API.
///
/// Example:
/// ```dart
/// final client = GeminiOpenAIClient(apiKey: 'your-gemini-api-key');
///
/// final response = await client.chat.completions.create(
///   ChatCompletionCreateRequest(
///     model: 'gemini-2.5-flash',
///     messages: [ChatMessage.user('Hello!')],
///   ),
/// );
/// ```
class GeminiOpenAIClient extends OpenAIClient {
  late final gai.GoogleAIClient _geminiClient;
  final http.Client? _ownHttpClient;
  late final http.Client _resourceHttpClient;

  final String _apiKey;
  final String _baseUrl;
  final gai.ApiVersion _apiVersion;

  /// The Gemini API key.
  String get apiKey => _apiKey;

  /// The base URL for the Gemini API.
  String get baseUrl => _baseUrl;

  /// The API version used for Gemini requests.
  gai.ApiVersion get apiVersion => _apiVersion;

  /// Thought signatures accumulated across calls.
  ///
  /// Maps tool call IDs to base64-encoded thought signatures that must be
  /// preserved for Gemini 3+ models. These are accumulated across multiple
  /// `create()` and `createStream()` calls.
  Map<String, String> thoughtSignatures = {};

  /// The resource name of a cached content to use for subsequent requests.
  ///
  /// Set this to a cached content resource name (e.g., `cachedContents/abc123`)
  /// to enable Gemini context caching. The cached content is prepended to each
  /// request's contents.
  ///
  /// Create cached content via [cachedContents], then assign the returned
  /// name here:
  /// ```dart
  /// final cached = await client.cachedContents.create(
  ///   CachedContent(
  ///     model: 'models/gemini-2.5-flash',
  ///     systemInstruction: Content(parts: [TextPart('You are helpful.')]),
  ///     contents: [...],
  ///     ttl: '3600s',
  ///   ),
  /// );
  /// client.cachedContent = cached.name;
  /// ```
  String? cachedContent;

  /// Access to the underlying Gemini cached contents API for creating,
  /// listing, updating, and deleting cached content resources.
  CachedContentsResource get cachedContents =>
      _geminiClient.cachedContents;

  /// Creates a new GeminiOpenAIClient.
  ///
  /// Parameters:
  /// - [apiKey]: Your Gemini API key.
  /// - [baseUrl]: Optional custom base URL for the Gemini API.
  /// - [apiVersion]: The API version to use (default: v1beta).
  /// - [client]: Optional custom HTTP client.
  GeminiOpenAIClient({
    required String apiKey,
    String baseUrl = 'https://generativelanguage.googleapis.com',
    gai.ApiVersion apiVersion = gai.ApiVersion.v1beta,
    http.Client? client,
  })  : _apiKey = apiKey,
        _baseUrl = baseUrl,
        _apiVersion = apiVersion,
        _ownHttpClient = client,
        super(httpClient: client) {
    _resourceHttpClient = client ?? http.Client();
    _geminiClient = _buildGeminiClient();
  }

  gai.GoogleAIClient _buildGeminiClient() {
    return gai.GoogleAIClient(
      config: gai.GoogleAIConfig(
        authProvider:
            _apiKey.isNotEmpty ? gai.ApiKeyProvider(_apiKey) : null,
        baseUrl: _baseUrl,
        apiVersion: _apiVersion,
      ),
      httpClient: _ownHttpClient,
    );
  }

  // ============================================================================
  // Override chat resource to route through Gemini
  // ============================================================================

  _GeminiChatResource? _geminiChat;

  @override
  ChatResource get chat => _geminiChat ??= _GeminiChatResource(
        geminiClient: _geminiClient,
        owner: this,
        // These base resource fields are required by the parent class but unused
        // since our overridden create()/createStream() bypass OpenAI's HTTP
        // pipeline.
        config: config,
        httpClient: _resourceHttpClient,
        interceptorChain: interceptorChain,
        requestBuilder: RequestBuilder(config: config),
      );

  /// Clears accumulated conversation state (thought signatures).
  ///
  /// Call this when starting a new conversation to avoid carrying over
  /// thought signatures from previous conversations.
  void clearConversationState() {
    thoughtSignatures = {};
  }

  /// Closes the underlying Gemini client.
  @override
  void close() {
    _geminiClient.close();
    if (_ownHttpClient == null) {
      _resourceHttpClient.close();
    }
    super.close();
  }
}

// ============================================================================
// Internal resource classes to intercept chat completions
// ============================================================================

class _GeminiChatResource extends ChatResource {
  final gai.GoogleAIClient geminiClient;
  final GeminiOpenAIClient owner;

  _GeminiChatResource({
    required this.geminiClient,
    required this.owner,
    required super.config,
    required super.httpClient,
    required super.interceptorChain,
    required super.requestBuilder,
  });

  _GeminiChatCompletionsResource? _geminiCompletions;

  @override
  ChatCompletionsResource get completions =>
      _geminiCompletions ??= _GeminiChatCompletionsResource(
        geminiClient: geminiClient,
        owner: owner,
        config: config,
        httpClient: httpClient,
        interceptorChain: interceptorChain,
        requestBuilder: requestBuilder,
      );
}

class _GeminiChatCompletionsResource extends ChatCompletionsResource {
  final gai.GoogleAIClient geminiClient;
  final GeminiOpenAIClient owner;

  _GeminiChatCompletionsResource({
    required this.geminiClient,
    required this.owner,
    required super.config,
    required super.httpClient,
    required super.interceptorChain,
    required super.requestBuilder,
  });

  @override
  Future<ChatCompletion> create(
    ChatCompletionCreateRequest request, {
    Future<void>? abortTrigger,
  }) async {
    final requestModel = request.model;

    // Convert messages.
    final messageResult = MessageContentConverter.toGemini(
      request.messages,
      thoughtSignatures: owner.thoughtSignatures,
    );

    // Build Gemini request.
    final geminiRequest = gai.GenerateContentRequest(
      contents: messageResult.contents,
      systemInstruction: messageResult.systemInstruction,
      tools: ChatCompletionRequestConverter.buildTools(request),
      toolConfig:
          ChatCompletionRequestConverter.buildToolConfig(request)?.toJson(),
      generationConfig:
          ChatCompletionRequestConverter.buildGenerationConfig(request),
      cachedContent: owner.cachedContent,
    );

    // Call Gemini API.
    final geminiResponse = await geminiClient.models.generateContent(
      model: requestModel,
      request: geminiRequest,
      abortTrigger: abortTrigger,
    );

    // Convert response.
    final result = ChatCompletionResponseConverter.convert(
      geminiResponse,
      model: requestModel,
    );

    // Accumulate thought signatures.
    owner.thoughtSignatures.addAll(result.thoughtSignatures);

    return result.completion;
  }

  @override
  Stream<ChatStreamEvent> createStream(
    ChatCompletionCreateRequest request, {
    Future<void>? abortTrigger,
  }) {
    final requestModel = request.model;

    // Convert messages.
    final messageResult = MessageContentConverter.toGemini(
      request.messages,
      thoughtSignatures: owner.thoughtSignatures,
    );

    // Build Gemini request.
    final geminiRequest = gai.GenerateContentRequest(
      contents: messageResult.contents,
      systemInstruction: messageResult.systemInstruction,
      tools: ChatCompletionRequestConverter.buildTools(request),
      toolConfig:
          ChatCompletionRequestConverter.buildToolConfig(request)?.toJson(),
      generationConfig:
          ChatCompletionRequestConverter.buildGenerationConfig(request),
      cachedContent: owner.cachedContent,
    );

    // Stream from Gemini API.
    final geminiStream = geminiClient.models.streamGenerateContent(
      model: requestModel,
      request: geminiRequest,
      abortTrigger: abortTrigger,
    );

    // Convert the Gemini stream to OpenAI format.
    final result = convertGeminiStream(
      geminiStream,
      model: requestModel,
    );

    // Accumulate thought signatures when the stream completes.
    result.thoughtSignatures.then((sigs) {
      owner.thoughtSignatures.addAll(sigs);
    });

    return result.events;
  }
}
