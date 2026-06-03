// Probe: send a 10k-token prefix to gemini-3.5-flash twice and dump the raw
// usageMetadata. Tests whether implicit caching engages and what field
// names the API actually returns (vs gemini-3-flash-preview which works
// in the bench).
//
// Usage: GEMINI_API_KEY=... dart run tool/probe_3_5_flash_cache.dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';

String _systemPromptOfSize(int sections) {
  final buf = StringBuffer();
  buf.writeln('You are an expert assistant with deep knowledge of many topics.');
  for (int i = 0; i < sections; i++) {
    buf.writeln('## Section $i: Knowledge fact');
    buf.writeln('Parameter ${i}_alpha is ${i * 17 + 42}. Component ${i}_beta calibration: ${i * 31 + 7}.');
    buf.writeln('Frequency: ${i * 3.14159} Hz. Tolerance: ${1.0 / (i + 1)}%.');
    buf.writeln('Temperature: ${(i * 0.0023) + 20.0} degrees. Pressure: ${(i * 1.5) + 100} kPa.');
    buf.writeln('Section ${i + 1} cross-references section ${i + 2}.');
    buf.writeln();
  }
  return buf.toString();
}

String? _apiKey() {
  final env = Platform.environment['GEMINI_API_KEY'];
  if (env != null && env.isNotEmpty) return env;
  final envFile = File('/Users/saadardati/IdeaProjects/llmcouncil/server/.env');
  if (!envFile.existsSync()) return null;
  for (final line in envFile.readAsLinesSync()) {
    if (line.startsWith('GEMINI_API_KEY=')) {
      var v = line.substring('GEMINI_API_KEY='.length).trim();
      if (v.startsWith('"') && v.endsWith('"')) v = v.substring(1, v.length - 1);
      return v;
    }
  }
  return null;
}

Future<Map<String, dynamic>> _generateContent({
  required String apiKey,
  required String model,
  required String systemPrompt,
  required String userMessage,
  bool withTools = false,
}) async {
  final client = HttpClient();
  try {
    // OpenAI-compat endpoint — same shape the bench's agent_kit OpenAIClient
    // ends up POSTing to via `chat.completions`. Cache-token extraction
    // happens server-side from Gemini's native usage metadata into the
    // OpenAI-shaped `usage.prompt_tokens_details.cached_tokens` field.
    final uri = Uri.parse(
      'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    );
    final tools = !withTools
        ? null
        : List<Map<String, dynamic>>.generate(
            10,
            (i) => {
              'type': 'function',
              'function': {
                'name': 'probe_tool_$i',
                'description': 'Filler tool $i so prefix bytes include a tools array.',
                'parameters': {
                  'type': 'object',
                  'additionalProperties': false,
                  'properties': {
                    'arg': {'type': 'string'},
                  },
                },
              },
            },
          );
    final body = jsonEncode({
      'model': model,
      'messages': [
        {'role': 'system', 'content': systemPrompt},
        {'role': 'user', 'content': userMessage},
      ],
      'max_tokens': 64,
      'tools': ?tools,
      if (tools != null) 'tool_choice': 'auto',
    });
    final req = await client.postUrl(uri);
    req.headers.set('content-type', 'application/json');
    req.headers.set('Authorization', 'Bearer $apiKey');
    final bytes = utf8.encode(body);
    req.contentLength = bytes.length;
    req.add(bytes);
    final resp = await req.close();
    final text = await resp.transform(utf8.decoder).join();
    if (resp.statusCode != 200) {
      stderr.writeln('HTTP ${resp.statusCode}: $text');
      return {};
    }
    return jsonDecode(text) as Map<String, dynamic>;
  } finally {
    client.close();
  }
}

Future<void> _probe({
  required String label,
  required String model,
  required String apiKey,
  required String systemPrompt,
  bool withTools = false,
}) async {
  print('--- $label ($model${withTools ? ' +tools' : ''}) ---');
  for (int turn = 1; turn <= 2; turn++) {
    final nonce = 'nonce_${DateTime.now().microsecondsSinceEpoch}_$turn';
    final result = await _generateContent(
      apiKey: apiKey,
      model: model,
      systemPrompt: systemPrompt,
      userMessage: 'Acknowledge $nonce. Reply in one word.',
      withTools: withTools,
    );
    // OpenAI-compat returns `usage`, not `usageMetadata`.
    final usage = result['usage'] as Map<String, dynamic>?;
    print('  turn $turn usage: $usage');
  }
  print('');
}

Future<void> main() async {
  final apiKey = _apiKey();
  if (apiKey == null) {
    stderr.writeln('No GEMINI_API_KEY available.');
    exit(1);
  }

  // Sweep system-prompt sizes so we can locate 3.5-flash's effective
  // minimum cacheable prefix. The bench fails for 3.5-flash at ~10k tokens
  // of prefix; my earlier probe succeeded at ~23k. Try 30/60/100/200
  // sections (≈ 7k / 14k / 23k / 45k tokens of system text alone).
  for (final sections in [30, 60, 100, 200]) {
    final systemPrompt = _systemPromptOfSize(sections);
    print('=== system prompt = $sections sections (${systemPrompt.length} chars) ===');
    await _probe(
      label: '3.5-flash with-tools',
      model: 'gemini-3.5-flash',
      apiKey: apiKey,
      systemPrompt: systemPrompt,
      withTools: true,
    );
    await _probe(
      label: '3-flash-preview with-tools',
      model: 'gemini-3-flash-preview',
      apiKey: apiKey,
      systemPrompt: systemPrompt,
      withTools: true,
    );
  }
}
