## Unreleased

- Update `googleai_dart` to `^13.0.0` and `openai_dart` to `^11.0.1`.
- Map `ReasoningEffort.max` to `ThinkingLevel.high` and `FinishReason.escalation` to `stop`.
- Require Dart `^3.13.0`.

## 0.3.0

- Upgrade dependencies.

## 0.2.0

- Initial release.
- Bidirectional conversion between OpenAI Chat Completions and Gemini API formats.
- Request conversion: `ChatCompletionCreateRequest` to Gemini `GenerateContentRequest` components.
- Response conversion: Gemini `GenerateContentResponse` to OpenAI `ChatCompletion`.
- Streaming: `GeminiStreamEventTransformer` and `convertGeminiStream()`.
- Tool calling: definitions, calls, results, and choice mapping.
- JSON Schema sanitization: `anyOf`/`oneOf`/`allOf`/`const` flattening.
- Thought signature preservation for Gemini 3+ models.
- Reasoning effort to thinking level mapping.
- Finish reason and usage metadata translation.
