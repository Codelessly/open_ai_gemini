# Gap Tracker: pi-mono Reference vs Our Implementation

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Complete

## Critical Priority

### Gap #1: SKIP_THOUGHT_SIGNATURE Sentinel for Gemini 3
- **Status:** [x]
- **Issue:** Gemini 3 requires `thoughtSignature` on ALL function calls when thinking mode is enabled. We passed `null` when no signature existed.
- **Fix:** Added `skipThoughtSignatureBytes` sentinel in `thought_signature_utils.dart`. Applied in `_convertAssistantMessage` when `isGemini3Model(modelId)` is true and no real signature exists.
- **Files:** `message_content_converter.dart`, `utils/thought_signature_utils.dart`

### Gap #2: Cross-Provider/Model Awareness for Thinking Blocks
- **Status:** [x]
- **Issue:** We always converted `reasoningContent` to `thought: true` regardless of source model.
- **Fix:** Added `sourceProvider`/`sourceModel` parameters. Only keeps thinking as `thought:true` when same provider AND model. Otherwise converts to plain text.
- **Files:** `message_content_converter.dart`

## High Priority

### Gap #3: Error/Success Distinction in Tool Results
- **Status:** [x]
- **Issue:** We always wrapped tool results as `{'result': content}`.
- **Fix:** Added `isError` parameter to `toGeminiFunctionResponse`. Uses `{'error': value}` for errors, `{'output': value}` for success.
- **Files:** `tool_mapper.dart`

### Gap #4: Unicode Surrogate Sanitization
- **Status:** [x]
- **Issue:** No sanitization of unicode surrogates before sending to Gemini API.
- **Fix:** Added `sanitizeSurrogates()` utility. Applied to all text content: system prompts, user messages, assistant text, and reasoning content.
- **Files:** `utils/sanitize_unicode.dart`, `message_content_converter.dart`

### Gap #5: Text Signature Round-Tripping
- **Status:** [x]
- **Issue:** We captured `__last_text__` signatures from responses but never re-injected them.
- **Fix:** When converting assistant messages back to Gemini, looks up `__last_text__` signature and attaches to text parts (only when same provider/model).
- **Files:** `message_content_converter.dart`

## Medium Priority

### Gap #6: Thought Signature Validation
- **Status:** [x]
- **Issue:** No base64 format validation on thought signatures.
- **Fix:** Added `isValidThoughtSignature()` and `resolveThoughtSignature()` utilities. Validates base64 format (length % 4 == 0, valid chars) and gates on same provider/model.
- **Files:** `utils/thought_signature_utils.dart`, `message_content_converter.dart`

### Gap #7: Multimodal Function Responses (Gemini 3)
- **Status:** [x]
- **Issue:** Tool results were text-only, no support for images.
- **Fix:** Added `imageData` and `modelId` parameters to `toGeminiFunctionResponse`. For Gemini 3: nests images inside `functionResponse.parts` using `FunctionResponseInlinePart`. For older models: `parts` is null (caller handles separately).
- **Files:** `tool_mapper.dart`

## Low Priority

### Gap #8: Missing Finish Reasons
- **Status:** [x] N/A
- **Issue:** The Dart SDK's `FinishReason` enum already matches all values we handle. No additional values exist in `googleai_dart` 3.6.0.

### Gap #9: Tool Call ID Normalization
- **Status:** [x]
- **Issue:** No stripping of special characters or length capping on tool call IDs.
- **Fix:** Added `normalizeToolCallId()` utility that replaces non-`[a-zA-Z0-9_-]` chars with `_` and caps at 64 chars. Controlled by `normalizeToolCallIds` parameter.
- **Files:** `utils/thought_signature_utils.dart`, `message_content_converter.dart`

### Gap #10: Empty Block Filtering
- **Status:** [x]
- **Issue:** Fallback to empty `TextPart('')` could cause issues with some models.
- **Fix:** Skip empty/whitespace-only text and reasoning content. Only use empty TextPart fallback when the entire message would be empty (last resort).
- **Files:** `message_content_converter.dart`

## Test Coverage

### Unit Tests: `test/gap_fixes_test.dart` (28 tests)
- Gap #1: 3 tests (sentinel on G3, no sentinel on non-G3, real sig preserved)
- Gap #2: 3 tests (same model, different model, no provider info)
- Gap #3: 3 tests (success key, error key, round-trip)
- Gap #4: 4 tests (user text, system, assistant, clean passthrough)
- Gap #5: 2 tests (injection, no injection for different model)
- Gap #6: 3 tests (valid base64, invalid base64, cross-provider)
- Gap #7: 2 tests (Gemini 3 inline parts, non-G3 no parts)
- Gap #9: 3 tests (special chars stripped, length capped, clean passthrough)
- Gap #10: 3 tests (empty text filtered, whitespace reasoning filtered, valid fallback)
- OpenAI round-trip: 2 tests (full conversation, argument preservation)

### E2E Tests: `test/gap_fixes_e2e_test.dart` (6 tests)
- Multi-turn tool calling with thought signatures (4 turns)
- Error tool results communicated to model
- Unicode CJK content round-trip
- GeminiOpenAIClient drop-in multi-turn
- Cross-provider OpenAI → Gemini context preservation
- Streaming with signature capture
