# open_ai_gemini

A bidirectional translation layer between the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) and [Google's Gemini API](https://ai.google.dev/api). Store all LLM conversations in one OpenAI-compliant format, regardless of the underlying provider.

## Features

- **Request conversion** - `ChatCompletionCreateRequest` &rarr; Gemini `GenerateContentRequest` components
- **Response conversion** - Gemini `GenerateContentResponse` &rarr; OpenAI `ChatCompletion`
- **Streaming** - Gemini streaming chunks &rarr; OpenAI `ChatStreamEvent`s
- **Tool calling** - Bidirectional tool definitions, tool calls, and tool results with automatic JSON Schema sanitization (`anyOf`/`oneOf`/`allOf` flattening)
- **Thought signatures** - Preserves Gemini 3+ cryptographic thought signatures across multi-turn tool-calling sequences
- **Thinking/reasoning** - Maps OpenAI `reasoning_effort` &harr; Gemini `thinkingConfig.thinkingLevel` and surfaces thinking tokens as `reasoningContent`
- **Cross-provider interop** - Maintain a single conversation history and freely switch between OpenAI and Gemini models mid-conversation

## Installation

```yaml
dependencies:
  open_ai_gemini: ^0.1.0
```

## Quick Start

### Convert an OpenAI request to Gemini format

```dart
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;
import 'package:googleai_dart/googleai_dart.dart' as gai;

// Build your request in OpenAI format.
final request = oai.ChatCompletionCreateRequest(
  model: 'gemini-3-flash-preview',
  messages: [
    oai.ChatMessage.system('You are a helpful assistant.'),
    oai.ChatMessage.user('What is the capital of France?'),
  ],
  tools: [
    oai.Tool.function(
      name: 'lookup_capital',
      description: 'Look up the capital of a country.',
      parameters: {
        'type': 'object',
        'properties': {
          'country': {'type': 'string'},
        },
        'required': ['country'],
      },
    ),
  ],
);

// Convert to Gemini format.
final gemini = ChatCompletionRequestConverter.convert(request);

// Send to Gemini API.
final response = await geminiClient.models.generateContent(
  model: 'gemini-3-flash-preview',
  request: gai.GenerateContentRequest(
    contents: gemini.contents,
    systemInstruction: gemini.systemInstruction,
    tools: gemini.tools,
    toolConfig: gemini.toolConfig?.toJson(),
    generationConfig: gemini.generationConfig,
  ),
);
```

### Convert a Gemini response back to OpenAI format

```dart
final result = ChatCompletionResponseConverter.convert(
  response,
  model: 'gemini-3-flash-preview',
);

// Standard OpenAI ChatCompletion object.
final completion = result.completion;
print(completion.choices.first.message.content);

// Preserve thought signatures for the next turn (Gemini 3+).
final thoughtSignatures = result.thoughtSignatures;
```

### Thought signature round-tripping (Gemini 3+)

Gemini 3 attaches cryptographic thought signatures to function calls that **must** be returned in subsequent turns. This package handles this automatically:

```dart
// Accumulate signatures across turns.
var signatures = <String, String>{};

// Turn 1: Get response with tool calls.
final result1 = ChatCompletionResponseConverter.convert(response1, model: model);
signatures = {...signatures, ...result1.thoughtSignatures};

// Store the assistant message in your OpenAI-format history.
history.add(result1.completion.choices.first.message);
history.add(oai.ChatMessage.tool(toolCallId: '...', content: '...'));

// Turn 2: Pass signatures back when converting the next request.
final gemini2 = ChatCompletionRequestConverter.convert(
  nextRequest,
  thoughtSignatures: signatures, // Re-injected into FunctionCallParts
);
```

### Streaming

```dart
// Using the transformer class.
final transformer = GeminiStreamEventTransformer(model: 'gemini-3-flash-preview');
final openAIStream = geminiStream.transform(transformer);

await for (final event in openAIStream) {
  final delta = event.choices?.first.delta;
  if (delta?.content != null) print(delta!.content);
}

// Or using the convenience function (also captures thought signatures).
final result = convertGeminiStream(geminiStream, model: 'gemini-3-flash-preview');
await for (final event in result.events) { /* ... */ }
final signatures = await result.thoughtSignatures;
```

### Cross-provider conversations

The primary use case: store all conversations in OpenAI format and freely switch providers.

```dart
final history = <oai.ChatMessage>[
  oai.ChatMessage.system('You are a geography expert.'),
];

// Turn 1: Use GPT.
history.add(oai.ChatMessage.user('What continent is France in?'));
final gptResponse = await openAIClient.chat.completions.create(
  oai.ChatCompletionCreateRequest(model: 'gpt-4.1-nano', messages: history),
);
history.add(gptResponse.choices.first.message);

// Turn 2: Switch to Gemini seamlessly.
history.add(oai.ChatMessage.user('And Japan?'));
final geminiReq = ChatCompletionRequestConverter.convert(
  oai.ChatCompletionCreateRequest(model: 'gemini-3-flash-preview', messages: history),
);
// ... send to Gemini, convert response, add to history.

// Turn 3: Switch back to GPT - full context preserved.
history.add(oai.ChatMessage.user('Which did we discuss first?'));
final gptResponse2 = await openAIClient.chat.completions.create(
  oai.ChatCompletionCreateRequest(model: 'gpt-4.1-nano', messages: history),
);
```

## API Reference

### Converters

| Class                             | Description                                                             |
|-----------------------------------|-------------------------------------------------------------------------|
| `ChatCompletionRequestConverter`  | Converts `ChatCompletionCreateRequest` &rarr; Gemini request components |
| `MessageContentConverter`         | Bidirectional message/content conversion                                |
| `ChatCompletionResponseConverter` | Converts `GenerateContentResponse` &rarr; `ChatCompletion`              |
| `GeminiStreamEventTransformer`    | `StreamTransformer` for streaming responses                             |
| `convertGeminiStream()`           | Convenience function for streaming with signature capture               |

### Mappers

| Class                | Description                                                          |
|----------------------|----------------------------------------------------------------------|
| `ToolMapper`         | Tool definitions, tool choice, tool results, and schema sanitization |
| `FinishReasonMapper` | Maps Gemini finish reasons to OpenAI equivalents                     |

### Translation reference

| OpenAI                            | Gemini                                         | Notes                                          |
|-----------------------------------|------------------------------------------------|------------------------------------------------|
| `role: "system"`                  | `systemInstruction`                            | Extracted from messages to top-level field     |
| `role: "assistant"`               | `role: "model"`                                | Direct rename                                  |
| `role: "tool"`                    | `role: "user"` + `FunctionResponsePart`        | Role changes; consecutive tool messages merged |
| `max_tokens`                      | `generationConfig.maxOutputTokens`             |                                                |
| `stop`                            | `generationConfig.stopSequences`               |                                                |
| `tools`                           | `tools[{functionDeclarations}]`                | Schemas sanitized for Gemini compatibility     |
| `tool_choice: "auto"`             | `functionCallingConfig.mode: AUTO`             |                                                |
| `tool_choice: "required"`         | `functionCallingConfig.mode: ANY`              |                                                |
| `tool_choice: "none"`             | `functionCallingConfig.mode: NONE`             |                                                |
| `response_format (json)`          | `responseMimeType: "application/json"`         |                                                |
| `reasoning_effort`                | `thinkingConfig.thinkingLevel`                 | `low`/`medium`/`high` mapped directly          |
| `finish_reason: "stop"`           | `STOP`                                         |                                                |
| `finish_reason: "length"`         | `MAX_TOKENS`                                   |                                                |
| `finish_reason: "content_filter"` | `SAFETY` / `RECITATION` / `BLOCKLIST` / `SPII` | Multiple Gemini reasons map to one             |
| `finish_reason: "tool_calls"`     | (tool calls detected)                          | Inferred from response content                 |

### Schema sanitization

Gemini rejects standard JSON Schema combinators. `ToolMapper.sanitizeSchema()` automatically handles:

- `anyOf` / `oneOf` with `const` values &rarr; `enum` array
- `anyOf` / `oneOf` with object variants &rarr; merged `properties`
- `allOf` &rarr; merged into single schema
- `const` &rarr; single-value `enum`
- Strips `strict`, `additionalProperties`, `$schema`

## License

MIT
