import 'dart:convert';

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
