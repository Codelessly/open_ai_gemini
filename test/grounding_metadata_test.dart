// Verifies the Gemini stream transformer surfaces native grounding sources
// (google_search + url_context) onto the OpenAI-compatible chunk JSON.
//
// Barrier: openai_dart's `ChatStreamEvent` has no annotations field and its
// `toJson()` is a CLOSED map literal — a `fromJson({...toJson(), key})` round
// trip would silently DROP an injected top-level key. So the transformer must
// emit a `ChatStreamEvent` SUBCLASS whose `toJson()` adds the custom
// `web_search_results` key. This test pins both the subclass-survival
// guarantee and the field extraction from `groundingMetadata.groundingChunks`.

import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:test/test.dart';

gai.GenerateContentResponse _groundedChunk() {
  return gai.GenerateContentResponse(
    candidates: [
      gai.Candidate(
        content: const gai.Content(
          role: 'model',
          parts: [gai.TextPart('Quantum computing uses qubits.')],
        ),
        finishReason: gai.FinishReason.stop,
        groundingMetadata: const gai.GroundingMetadata(
          groundingChunks: [
            gai.GroundingChunk(
              web: gai.Web(
                uri: 'https://example.com/qubits',
                title: 'Qubits 101',
              ),
            ),
            gai.GroundingChunk(
              web: gai.Web(
                uri: 'https://news.example.org/quantum',
                title: 'Quantum News',
              ),
            ),
            // A non-web grounding chunk (no uri) must be skipped.
            gai.GroundingChunk(),
          ],
        ),
      ),
    ],
  );
}

void main() {
  group('Gemini grounding → web_search_results', () {
    test('surfaces groundingChunks[].web{uri,title} on chunk toJson()', () async {
      final result = convertGeminiStream(
        Stream.value(_groundedChunk()),
        model: 'gemini-3-flash-preview',
      );

      final events = await result.events.toList();

      // Find the event carrying the grounding sources.
      final grounded = events.map((e) => e.toJson()).where((j) => j.containsKey('web_search_results')).toList();

      expect(grounded, hasLength(1), reason: 'sources emitted exactly once');
      expect(grounded.single['web_search_results'], [
        {'url': 'https://example.com/qubits', 'title': 'Qubits 101'},
        {'url': 'https://news.example.org/quantum', 'title': 'Quantum News'},
      ]);
    });

    test('emits NOTHING extra when there is no groundingMetadata', () async {
      final plain = gai.GenerateContentResponse(
        candidates: [
          gai.Candidate(
            content: const gai.Content(
              role: 'model',
              parts: [gai.TextPart('hello')],
            ),
            finishReason: gai.FinishReason.stop,
          ),
        ],
      );

      final result = convertGeminiStream(
        Stream.value(plain),
        model: 'gemini-3-flash-preview',
      );
      final events = await result.events.toList();

      final anyGrounded = events.any(
        (e) => e.toJson().containsKey('web_search_results'),
      );
      expect(anyGrounded, isFalse);
      // Plain chunks remain ordinary ChatStreamEvents.
      expect(events, everyElement(isA<oai.ChatStreamEvent>()));
    });
  });
}
