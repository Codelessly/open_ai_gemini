/// Represents binary media data from a Gemini response that cannot be
/// represented in OpenAI's `AssistantMessage.content` string.
///
/// Since OpenAI's assistant messages only support text content, any images,
/// audio, or other binary data returned by Gemini models (e.g., from image
/// generation) are captured here for the caller to handle.
///
/// Media attachments are also accepted when converting OpenAI → Gemini, so
/// they round-trip correctly through the translation layer.
class MediaAttachment {
  /// The MIME type of the media (e.g., `image/png`, `audio/wav`).
  final String mimeType;

  /// The base64-encoded binary data, or `null` if [fileUri] is set.
  final String? data;

  /// A file URI (HTTP, HTTPS, or GCS `gs://` URI), or `null` if [data] is set.
  final String? fileUri;

  /// Creates an inline media attachment with base64-encoded data.
  const MediaAttachment.inline({
    required this.mimeType,
    required String this.data,
  }) : fileUri = null;

  /// Creates a file-referenced media attachment.
  const MediaAttachment.file({
    required this.mimeType,
    required String this.fileUri,
  }) : data = null;

  const MediaAttachment._({
    required this.mimeType,
    this.data,
    this.fileUri,
  });

  /// Whether this attachment contains inline base64 data.
  bool get isInline => data != null;

  /// Whether this attachment references a file URI.
  bool get isFile => fileUri != null;

  /// Whether this is an image based on MIME type.
  bool get isImage => mimeType.startsWith('image/');

  /// Whether this is audio based on MIME type.
  bool get isAudio => mimeType.startsWith('audio/');

  /// Whether this is a video based on MIME type.
  bool get isVideo => mimeType.startsWith('video/');

  /// Whether this is a PDF.
  bool get isPdf => mimeType == 'application/pdf';

  /// Returns a data URL representation (`data:<mime>;base64,<data>`).
  ///
  /// Only valid for inline attachments. Throws if [data] is null.
  String toDataUrl() {
    if (data == null) {
      throw StateError('Cannot create data URL for file-referenced attachment.');
    }
    return 'data:$mimeType;base64,$data';
  }

  /// Serializes to a JSON-compatible map for storage.
  Map<String, dynamic> toJson() => {
    'mimeType': mimeType,
    if (data != null) 'data': data,
    if (fileUri != null) 'fileUri': fileUri,
  };

  /// Deserializes from a JSON map.
  factory MediaAttachment.fromJson(Map<String, dynamic> json) {
    return MediaAttachment._(
      mimeType: json['mimeType'] as String,
      data: json['data'] as String?,
      fileUri: json['fileUri'] as String?,
    );
  }

  @override
  String toString() => isInline
      ? 'MediaAttachment.inline($mimeType, ${data!.length} chars)'
      : 'MediaAttachment.file($mimeType, $fileUri)';
}
