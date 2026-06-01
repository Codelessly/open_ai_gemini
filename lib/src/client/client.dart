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
import '../utils/thought_signature_utils.dart';

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
  /// Maps tool call IDs to base64-encoded thought signatures captured from
  /// Gemini responses. These are accumulated across multiple `create()` and
  /// `createStream()` calls within a single conversation.
  ///
  /// The map's lifecycle is bound to the conversation, not to the client
  /// instance — [clearConversationState] wipes it. Cross-conversation
  /// persistence happens through a different mechanism: thought signatures
  /// are also encoded directly into the OpenAI `tool_call.id` field
  /// (`tsig_<base64Url>__<originalId>`) when responses are converted, so
  /// signatures survive any JSON round-trip through the caller's history
  /// store without needing the map at all.
  ///
  /// The map remains useful as a hot cache for in-flight conversion (e.g. an
  /// outgoing request that references the previous turn's tool call by its
  /// raw id) and as a fallback for tool-call ids that were generated outside
  /// our encoded form.
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

  /// When true, appends Gemini's native Google Search grounding tool to every
  /// request, letting the model ground its answers in live web results.
  ///
  /// This is sent as a separate native `gai.Tool` entry (it must NOT be passed
  /// through the OpenAI-compat tools array, which rejects `google_search`).
  bool enableGoogleSearch = false;

  /// When true, appends Gemini's native URL Context tool to every request,
  /// letting the model fetch and analyze content from URLs in the prompt.
  ///
  /// This is sent as a separate native `gai.Tool` entry (it must NOT be passed
  /// through the OpenAI-compat tools array, which rejects `url_context`).
  bool enableUrlContext = false;

  /// Access to the underlying Gemini cached contents API for creating,
  /// listing, updating, and deleting cached content resources.
  CachedContentsResource get cachedContents => _geminiClient.cachedContents;

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
  }) : _apiKey = apiKey,
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
        authProvider: _apiKey.isNotEmpty ? gai.ApiKeyProvider(_apiKey) : null,
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
  ///
  /// Cross-conversation persistence of signatures is handled separately:
  /// signatures are encoded into the OpenAI `tool_call.id` field when
  /// responses are converted, so they survive `assistantMessage.toJson()`
  /// → JSON store → `assistantMessage.fromJson()` round-trips. The map is a
  /// hot cache, not the source of truth — wiping it is safe and correct.
  void clearConversationState() {
    thoughtSignatures = {};
  }

  /// Alias for [clearConversationState], retained as an explicit reset
  /// escape hatch with a name that focuses on the signature subset.
  void clearThoughtSignatures() {
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
  ChatCompletionsResource get completions => _geminiCompletions ??= _GeminiChatCompletionsResource(
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

    // Convert messages. Pass the model ID for Gemini 3 sentinel support,
    // and tag the source as ('gemini', requestModel) so any accumulated
    // thought signatures (captured from this same client's previous Gemini
    // responses) actually get re-injected into outgoing FunctionCallParts.
    // Without sourceProvider/sourceModel, the converter treats every
    // assistant message as cross-provider and drops all real signatures,
    // leaving only the sentinel fallback — which Gemini 3+ rejects with
    // "Function call is missing a thought_signature" in many turn shapes.
    final messageResult = MessageContentConverter.toGemini(
      request.messages,
      thoughtSignatures: owner.thoughtSignatures,
      modelId: requestModel,
      sourceProvider: 'gemini',
      sourceModel: requestModel,
    );

    // Build Gemini request.
    final geminiRequest = gai.GenerateContentRequest(
      contents: messageResult.contents,
      systemInstruction: messageResult.systemInstruction,
      tools: ChatCompletionRequestConverter.appendGroundingTools(
        ChatCompletionRequestConverter.buildTools(request),
        enableGoogleSearch: owner.enableGoogleSearch,
        enableUrlContext: owner.enableUrlContext,
      ),
      toolConfig: ChatCompletionRequestConverter.buildToolConfig(request),
      generationConfig: ChatCompletionRequestConverter.buildGenerationConfig(request),
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

    // Convert messages. Pass the model ID for Gemini 3 sentinel support,
    // and tag the source as ('gemini', requestModel) so any accumulated
    // thought signatures (captured from this same client's previous Gemini
    // responses) actually get re-injected into outgoing FunctionCallParts.
    // Without sourceProvider/sourceModel, the converter treats every
    // assistant message as cross-provider and drops all real signatures,
    // leaving only the sentinel fallback — which Gemini 3+ rejects with
    // "Function call is missing a thought_signature" in many turn shapes.
    final messageResult = MessageContentConverter.toGemini(
      request.messages,
      thoughtSignatures: owner.thoughtSignatures,
      modelId: requestModel,
      sourceProvider: 'gemini',
      sourceModel: requestModel,
    );

    // Build Gemini request.
    final geminiRequest = gai.GenerateContentRequest(
      contents: messageResult.contents,
      systemInstruction: messageResult.systemInstruction,
      tools: ChatCompletionRequestConverter.appendGroundingTools(
        ChatCompletionRequestConverter.buildTools(request),
        enableGoogleSearch: owner.enableGoogleSearch,
        enableUrlContext: owner.enableUrlContext,
      ),
      toolConfig: ChatCompletionRequestConverter.buildToolConfig(request),
      generationConfig: ChatCompletionRequestConverter.buildGenerationConfig(request),
      cachedContent: owner.cachedContent,
    );

    // Stream from Gemini API.
    final geminiStream = geminiClient.models.streamGenerateContent(
      model: requestModel,
      request: geminiRequest,
      abortTrigger: abortTrigger,
    );

    // Convert the Gemini stream to OpenAI format.
    //
    // `convertGeminiStream` populates its internal signature map as the
    // underlying Gemini stream emits chunks. We tap into the returned event
    // stream and synchronously mirror any signature that came along with each
    // emitted tool_call into `owner.thoughtSignatures` BEFORE the event
    // reaches the downstream consumer. This eliminates the previous race
    // where the signatures Future's `.then(...)` could resolve AFTER the
    // upstream tool-call loop had already kicked off the NEXT request, which
    // would then read an empty map.
    //
    // Tool-call signatures are ALSO baked into the emitted tool_call.id via
    // the `tsig_…__` encoding, so even consumers that never read
    // `owner.thoughtSignatures` see the signatures travel with the message.
    // The map remains useful for text signatures (`__last_text__`) and for
    // ids generated outside our encoded form.
    final result = convertGeminiStream(
      geminiStream,
      model: requestModel,
    );

    final tappedEvents = result.events.map((event) {
      final choices = event.choices;
      if (choices == null) return event;
      for (final choice in choices) {
        final toolCallDeltas = choice.delta.toolCalls;
        if (toolCallDeltas == null) continue;
        for (final delta in toolCallDeltas) {
          final id = delta.id;
          if (id == null) continue;
          final decoded = decodeThoughtSignatureFromToolCallId(id);
          if (decoded.signatureBase64 != null) {
            owner.thoughtSignatures[id] = decoded.signatureBase64!;
          }
        }
      }
      return event;
    });

    // Keep the post-completion drain too, so text signatures
    // (`__last_text__`, which don't have a tool_call.id to encode into) and
    // any other map entries still land — but do it in a way that's purely
    // additive. This is no longer correctness-critical for tool calls.
    result.thoughtSignatures.then((sigs) {
      owner.thoughtSignatures.addAll(sigs);
    });

    return tappedEvents;
  }
}
