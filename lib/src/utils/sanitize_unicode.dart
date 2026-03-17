/// Sanitizes text by replacing lone Unicode surrogates with the Unicode
/// replacement character (U+FFFD).
///
/// Lone surrogates (U+D800–U+DFFF) are invalid in UTF-8 and JSON. The Gemini
/// API will reject them. This function makes text safe for API transmission
/// while preserving all valid content including valid surrogate pairs (which
/// represent supplementary plane characters).
String sanitizeSurrogates(String text) {
  // In Dart, strings are UTF-16. Lone surrogates are code units in the
  // range 0xD800–0xDFFF that don't form a valid pair.
  final buffer = StringBuffer();
  for (var i = 0; i < text.length; i++) {
    final code = text.codeUnitAt(i);
    if (code >= 0xD800 && code <= 0xDBFF) {
      // High surrogate — check for valid pair.
      if (i + 1 < text.length) {
        final next = text.codeUnitAt(i + 1);
        if (next >= 0xDC00 && next <= 0xDFFF) {
          // Valid surrogate pair — keep both.
          buffer.writeCharCode(code);
          buffer.writeCharCode(next);
          i++; // Skip the low surrogate.
          continue;
        }
      }
      // Lone high surrogate → replace.
      buffer.write('\uFFFD');
    } else if (code >= 0xDC00 && code <= 0xDFFF) {
      // Lone low surrogate → replace.
      buffer.write('\uFFFD');
    } else {
      buffer.writeCharCode(code);
    }
  }
  return buffer.toString();
}
