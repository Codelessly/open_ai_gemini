import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:open_ai_gemini/open_ai_gemini.dart';
import 'package:openai_dart/openai_dart.dart' as oai;

void main() async {
  final geminiClient = gai.GoogleAIClient.withApiKey('YOUR_API_KEY');
  const model = 'gemini-3-flash-preview';

  // Maintain conversation history in OpenAI format.
  final history = <oai.ChatMessage>[
    oai.ChatMessage.system('You are a helpful assistant.'),
    oai.ChatMessage.user('What is the capital of France?'),
  ];

  // Build an OpenAI-format request.
  final request = oai.ChatCompletionCreateRequest(
    model: model,
    messages: history,
  );

  // Convert to Gemini format.
  final geminiReq = ChatCompletionRequestConverter.convert(request);

  // Send to Gemini API.
  final response = await geminiClient.models.generateContent(
    model: model,
    request: gai.GenerateContentRequest(
      contents: geminiReq.contents,
      systemInstruction: geminiReq.systemInstruction,
      generationConfig: geminiReq.generationConfig,
    ),
  );

  // Convert response back to OpenAI format.
  final result = ChatCompletionResponseConverter.convert(
    response,
    model: model,
  );

  // Use like any OpenAI ChatCompletion.
  print(result.completion.choices.first.message.content);

  // Add to history for the next turn.
  history.add(result.completion.choices.first.message);
}
