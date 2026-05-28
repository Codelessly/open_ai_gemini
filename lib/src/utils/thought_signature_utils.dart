import 'dart:convert';

import 'package:openai_dart/openai_dart.dart' as oai;

/// The sentinel value that tells the Gemini API to skip thought signature
/// validation. Used for unsigned function call parts (e.g. replayed from
/// providers without thought signatures).
///
/// See: https://ai.google.dev/gemini-api/docs/thought-signatures
const String skipThoughtSignatureValidator = 'skip_thought_signature_validator';

/// Sentinel as bytes for injection into [FunctionCallPart.thoughtSignature].
final List<int> skipThoughtSignatureBytes = List.unmodifiable(utf8.encode(skipThoughtSignatureValidator));

/// Base64 pattern: only allows [A-Za-z0-9+/] with optional = padding.
final RegExp _base64Pattern = RegExp(r'^[A-Za-z0-9+/]+={0,2}$');

/// Returns `true` if [signature] is a well-formed base64 string.
///
/// Checks:
/// 1. Non-null and non-empty
/// 2. Length is a multiple of 4
/// 3. Matches the base64 character set
bool isValidThoughtSignature(String? signature) {
  if (signature == null || signature.isEmpty) return false;
  if (signature.length % 4 != 0) return false;
  return _base64Pattern.hasMatch(signature);
}

/// Returns the [signature] only if it is valid base64 AND comes from the same
/// provider and model. Otherwise returns `null`.
///
/// This prevents stale or cross-provider signatures from being sent to the
/// Gemini API, which would cause errors.
String? resolveThoughtSignature({
  required bool isSameProviderAndModel,
  required String? signature,
}) {
  if (!isSameProviderAndModel) return null;
  if (!isValidThoughtSignature(signature)) return null;
  return signature;
}

/// Returns `true` if [modelId] refers to a Gemini 3 model.
///
/// Gemini 3 models require `thoughtSignature` on ALL function call parts
/// when thinking mode is enabled.
bool isGemini3Model(String? modelId) {
  if (modelId == null) return false;
  return RegExp(r'gemini-3(?:\.\d+)?-').hasMatch(modelId.toLowerCase());
}

/// Normalizes a tool call ID by replacing non-alphanumeric characters
/// (except _ and -) with underscores and capping at 64 characters.
String normalizeToolCallId(String id) {
  final normalized = id.replaceAll(RegExp(r'[^a-zA-Z0-9_-]'), '_');
  if (normalized.length > 64) return normalized.substring(0, 64);
  return normalized;
}

// ---------------------------------------------------------------------------
// Tool-call-id signature encoding
// ---------------------------------------------------------------------------
//
// Gemini 3+ requires every replayed function_call part to carry the original
// `thoughtSignature` bytes the model emitted. The OpenAI `ChatMessage` JSON
// shape has no field for that, but it DOES have a `tool_call.id` field that
// survives `toJson()` / `fromJson()` round-trips through any opaque JSON
// store.
//
// We therefore piggyback the signature onto the id with a prefix:
//
//     tsig_<base64UrlNoPadding(signature)>__<originalId>
//
// The model never sees this id — Gemini's API takes the function name + args,
// not the OpenAI id — so it's safe to use as opaque protocol metadata. On the
// outgoing path we split it back out: the real Gemini `FunctionCall` gets the
// `originalId`, and the `FunctionCallPart` gets the decoded signature bytes.

/// Prefix that marks a tool-call id as carrying an embedded base64-url
/// thought signature.
const String _thoughtSignatureIdPrefix = 'tsig_';

/// Separator between the embedded signature and the original id.
const String _thoughtSignatureIdSeparator = '__';

/// Encodes [signatureBase64] (standard base64 from a previous Gemini response)
/// into a tool-call id alongside [originalId].
///
/// The returned id has the shape `tsig_<base64Url>__<originalId>` so it:
/// 1. Survives JSON round-trips through `oai.ToolCall.toJson()`/`fromJson()`.
/// 2. Stays within the typical id length limit and uses only URL-safe chars.
/// 3. Can be losslessly decoded by [decodeThoughtSignatureFromToolCallId].
String encodeThoughtSignatureInToolCallId({
  required String signatureBase64,
  required String originalId,
}) {
  // Re-encode standard base64 → base64Url (no padding) for id safety.
  final bytes = base64Decode(signatureBase64);
  final urlEncoded = base64UrlEncode(bytes).replaceAll('=', '');
  return '$_thoughtSignatureIdPrefix$urlEncoded$_thoughtSignatureIdSeparator$originalId';
}

/// Result of decoding a tool-call id that may carry an embedded thought
/// signature.
class DecodedToolCallId {
  /// Base64 (standard, padded) form of the embedded thought signature, or
  /// `null` if [encodedId] has no `tsig_` prefix.
  final String? signatureBase64;

  /// The original tool-call id with the `tsig_…__` prefix stripped (or the
  /// input id unchanged when there was no prefix).
  final String originalId;

  const DecodedToolCallId({
    required this.signatureBase64,
    required this.originalId,
  });
}

/// Splits a tool-call id of the form `tsig_<base64Url>__<originalId>` back into
/// its components.
///
/// If [encodedId] does not start with `tsig_` or is malformed, the input is
/// returned unchanged via [DecodedToolCallId.originalId] with a `null`
/// signature.
DecodedToolCallId decodeThoughtSignatureFromToolCallId(String encodedId) {
  if (!encodedId.startsWith(_thoughtSignatureIdPrefix)) {
    return DecodedToolCallId(signatureBase64: null, originalId: encodedId);
  }
  final afterPrefix = encodedId.substring(_thoughtSignatureIdPrefix.length);
  final sepIndex = afterPrefix.indexOf(_thoughtSignatureIdSeparator);
  if (sepIndex < 0) {
    // Malformed — treat as a plain id.
    return DecodedToolCallId(signatureBase64: null, originalId: encodedId);
  }
  final urlEncoded = afterPrefix.substring(0, sepIndex);
  final originalId = afterPrefix.substring(sepIndex + _thoughtSignatureIdSeparator.length);

  // Re-pad and decode the URL-safe base64 back to standard base64.
  final padNeeded = (4 - (urlEncoded.length % 4)) % 4;
  final padded = urlEncoded + ('=' * padNeeded);
  final List<int> bytes;
  try {
    bytes = base64Url.decode(padded);
  } catch (_) {
    // If decoding fails the id is just a coincidental `tsig_…__…` string
    // rather than something we encoded; fall back to treating it as plain.
    return DecodedToolCallId(signatureBase64: null, originalId: encodedId);
  }
  return DecodedToolCallId(
    signatureBase64: base64Encode(bytes),
    originalId: originalId,
  );
}

// ---------------------------------------------------------------------------
// Cross-provider sanitization
// ---------------------------------------------------------------------------
//
// The `tsig_<base64Sig>__<originalId>` encoding piggybacks the thought
// signature on the tool_call.id so it survives JSON round-trips through an
// arbitrary OpenAI-shaped history store. That works perfectly for replaying
// the history back to Gemini.
//
// However, non-Gemini providers (OpenAI, Anthropic, Azure, etc.) reject the
// long encoded id at the wire:
//
//   - OpenAI: tool_call.id is capped at 64 chars (40 for some Azure
//     deployments).
//   - Anthropic: tool_use_id has a stricter regex than OpenAI.
//   - Bedrock / others: similar length caps.
//
// The fix is symmetrical with what `MessageContentConverter.toGemini` does
// for the Gemini side: when building a request for a non-Gemini provider,
// strip the `tsig_<sig>__` prefix from every assistant `tool_call.id` and
// every tool `tool_call_id` so the wire payload only carries the short
// original id. The encoded id is still preserved in our in-memory history
// (any future Gemini replay re-reads it from there).
//
// See:
//   - https://github.com/BerriAI/litellm/pull/16895
//   - https://github.com/BerriAI/litellm/issues/18160
//   - https://github.com/openai/codex/issues/7519

/// Returns a deep copy of [messages] in which every assistant
/// `tool_call.id` and every tool `tool_call_id` of the form
/// `tsig_<base64Url>__<originalId>` has been collapsed back to just
/// `<originalId>`.
///
/// This MUST be called by the caller before sending a request to any
/// non-Gemini provider (OpenAI, Anthropic, Azure, Bedrock, ...) whose
/// conversation history may contain assistant turns that originally came
/// from a Gemini 3+ model. The encoded id is opaque protocol metadata that
/// only `MessageContentConverter.toGemini` knows how to consume; passing it
/// over the wire to a non-Gemini provider will fail with errors like:
///
///   Invalid 'messages[N].tool_calls[0].id': string too long.
///   Expected a string with maximum length 64, but got a string with
///   length 85 instead.
///
/// Messages that don't carry the `tsig_` prefix are returned unchanged
/// (same instance), so this is safe to call unconditionally on any
/// non-Gemini-bound request.
///
/// The returned list is always a fresh `List<oai.ChatMessage>`; the caller
/// is free to pass it directly into `ChatCompletionCreateRequest.messages`.
List<oai.ChatMessage> sanitizeMessagesForNonGeminiProvider(
  List<oai.ChatMessage> messages,
) {
  final sanitized = <oai.ChatMessage>[];
  for (final message in messages) {
    sanitized.add(_sanitizeMessage(message));
  }
  return sanitized;
}

oai.ChatMessage _sanitizeMessage(oai.ChatMessage message) {
  if (message is oai.AssistantMessage) {
    final toolCalls = message.toolCalls;
    if (toolCalls == null || toolCalls.isEmpty) return message;

    var anyChanged = false;
    final newToolCalls = <oai.ToolCall>[];
    for (final tc in toolCalls) {
      if (!tc.id.startsWith(_thoughtSignatureIdPrefix)) {
        newToolCalls.add(tc);
        continue;
      }
      final decoded = decodeThoughtSignatureFromToolCallId(tc.id);
      if (decoded.originalId == tc.id) {
        // Malformed `tsig_...` id we couldn't decode — keep as-is.
        newToolCalls.add(tc);
        continue;
      }
      anyChanged = true;
      newToolCalls.add(
        oai.ToolCall(
          id: decoded.originalId,
          type: tc.type,
          function: tc.function,
        ),
      );
    }
    if (!anyChanged) return message;
    return message.copyWith(toolCalls: newToolCalls);
  }

  if (message is oai.ToolMessage) {
    final id = message.toolCallId;
    if (!id.startsWith(_thoughtSignatureIdPrefix)) return message;
    final decoded = decodeThoughtSignatureFromToolCallId(id);
    if (decoded.originalId == id) return message;
    return message.copyWith(toolCallId: decoded.originalId);
  }

  return message;
}
