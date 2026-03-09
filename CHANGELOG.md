## 0.1.0

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
