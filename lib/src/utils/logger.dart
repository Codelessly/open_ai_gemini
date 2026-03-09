import 'dart:developer' as developer;

class GeminiOpenAILogger {
  const GeminiOpenAILogger._();

  static void warn(String message) {
    developer.log(message, name: 'open_ai_gemini', level: 900);
  }

  static void logUnsupportedParam(String paramName, dynamic value) {
    if (value != null) {
      warn('Unsupported OpenAI parameter "$paramName" will be ignored.');
    }
  }
}
