@Tags(['e2e'])
library;

import 'dart:io';

import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:test/test.dart';

/// Live reproduction of the Gemini 3.x mixed-tools 400:
///
///   ApiException(400): Please enable
///   tool_config.include_server_side_tool_invocations to use Built-in tools
///   with Function calling.
///
/// Seen in production on `gemini-3.1-pro-preview` in a court where the agent
/// carried both function tools (file/research/...) AND native grounding
/// (google_search + url_context). Without
/// `tool_config.includeServerSideToolInvocations`, the streaming request is
/// rejected. This drives the request through the REAL client path
/// ([GeminiOpenAIClient.createStream]) with grounding enabled + a function
/// tool and asserts the stream completes (no 400).
///
/// Requires GEMINI_API_KEY (environment or `.env`); skipped otherwise.
String? _geminiApiKey() {
  final fromEnv = Platform.environment['GEMINI_API_KEY'];
  if (fromEnv != null && fromEnv.isNotEmpty) return fromEnv;
  final envFile = File('.env');
  if (envFile.existsSync()) {
    for (final line in envFile.readAsLinesSync()) {
      final trimmed = line.trim();
      if (trimmed.startsWith('GEMINI_API_KEY=')) {
        final v = trimmed.substring('GEMINI_API_KEY='.length).trim();
        if (v.isNotEmpty) return v;
      }
    }
  }
  return null;
}

void main() {
  final apiKey = _geminiApiKey();

  group('Gemini native grounding + function tools (mixed tools)', () {
    test(
      'createStream on gemini-3.1-pro-preview does not 400 with grounding + a function tool',
      () async {
        final client = GeminiOpenAIClient(apiKey: apiKey!)
          ..enableGoogleSearch = true
          ..enableUrlContext = true;
        addTearDown(client.close);

        final request = oai.ChatCompletionCreateRequest(
          model: 'gemini-3.1-pro-preview',
          messages: [
            oai.ChatMessage.user(
              'What is the latest stable Dart SDK version? Use web search.',
            ),
          ],
          tools: [
            oai.Tool.function(
              name: 'file',
              description: 'Read a file from disk',
              parameters: const {
                'type': 'object',
                'properties': {
                  'path': {'type': 'string'},
                },
                'required': ['path'],
              },
            ),
          ],
          toolChoice: oai.ToolChoice.auto(),
        );

        // Before the fix this throws ApiException(400) mid-stream. After the
        // fix (includeServerSideToolInvocations: true) the stream completes.
        final events = await client.chat.completions.createStream(request).toList();
        expect(events, isNotEmpty);
      },
      timeout: const Timeout(Duration(minutes: 3)),
      skip: apiKey == null ? 'GEMINI_API_KEY not set' : false,
    );
  });
}
