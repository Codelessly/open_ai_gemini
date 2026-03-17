import 'dart:convert';

import 'package:googleai_dart/googleai_dart.dart' as gai;
import 'package:openai_dart/openai_dart.dart' as oai;

import '../models/media_attachment.dart';
import '../utils/thought_signature_utils.dart';

/// Maps tool definitions, tool calls, and tool choices between
/// OpenAI and Gemini formats.
class ToolMapper {
  const ToolMapper._();

  // ---------------------------------------------------------------------------
  // Tool Definitions: OpenAI → Gemini
  // ---------------------------------------------------------------------------

  /// Converts a list of OpenAI [oai.Tool] definitions to a Gemini [gai.Tool]
  /// containing all function declarations.
  static gai.Tool? toGeminiTools(List<oai.Tool>? tools) {
    if (tools == null || tools.isEmpty) return null;

    final declarations = <gai.FunctionDeclaration>[];
    for (final tool in tools) {
      declarations.add(
        gai.FunctionDeclaration(
          name: tool.function.name,
          description: tool.function.description ?? '',
          parameters: tool.function.parameters != null
              ? gai.Schema.fromJson(sanitizeSchema(tool.function.parameters!))
              : null,
        ),
      );
    }
    return gai.Tool(functionDeclarations: declarations);
  }

  /// Converts a list of OpenAI [oai.Tool] definitions to a list of
  /// [gai.FunctionDeclaration] objects.
  static List<gai.FunctionDeclaration>? toGeminiFunctionDeclarations(
    List<oai.Tool>? tools,
  ) {
    if (tools == null || tools.isEmpty) return null;
    return tools
        .map(
          (tool) => gai.FunctionDeclaration(
            name: tool.function.name,
            description: tool.function.description ?? '',
            parameters: tool.function.parameters != null
                ? gai.Schema.fromJson(
                    sanitizeSchema(tool.function.parameters!),
                  )
                : null,
          ),
        )
        .toList();
  }

  // ---------------------------------------------------------------------------
  // Tool Choice: OpenAI → Gemini
  // ---------------------------------------------------------------------------

  /// Converts an OpenAI [oai.ToolChoice] to a Gemini [gai.ToolConfig].
  static gai.ToolConfig? toGeminiToolConfig(oai.ToolChoice? toolChoice) {
    if (toolChoice == null) return null;

    return switch (toolChoice) {
      oai.ToolChoiceAuto() => gai.ToolConfig(
        functionCallingConfig: gai.FunctionCallingConfig(
          mode: gai.FunctionCallingMode.auto,
        ),
      ),
      oai.ToolChoiceRequired() => gai.ToolConfig(
        functionCallingConfig: gai.FunctionCallingConfig(
          mode: gai.FunctionCallingMode.any,
        ),
      ),
      oai.ToolChoiceNone() => gai.ToolConfig(
        functionCallingConfig: gai.FunctionCallingConfig(
          mode: gai.FunctionCallingMode.none,
        ),
      ),
      oai.ToolChoiceFunction(:final name) => gai.ToolConfig(
        functionCallingConfig: gai.FunctionCallingConfig(
          mode: gai.FunctionCallingMode.any,
          allowedFunctionNames: [name],
        ),
      ),
    };
  }

  // ---------------------------------------------------------------------------
  // Tool Calls: Gemini → OpenAI
  // ---------------------------------------------------------------------------

  /// Converts a Gemini [gai.FunctionCallPart] to an OpenAI [oai.ToolCall].
  ///
  /// [toolCallId] should be a unique identifier for this tool call.
  static oai.ToolCall toOpenAIToolCall(
    gai.FunctionCallPart part, {
    required String toolCallId,
  }) {
    return oai.ToolCall(
      id: toolCallId,
      type: 'function',
      function: oai.FunctionCall(
        name: part.functionCall.name,
        arguments: jsonEncode(part.functionCall.args ?? {}),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Tool Results: OpenAI → Gemini
  // ---------------------------------------------------------------------------

  /// Converts an OpenAI tool message content to a Gemini
  /// [gai.FunctionResponsePart].
  ///
  /// [functionName] is the name of the function that was called.
  /// [content] is the result string from the tool execution.
  /// [isError] when true, wraps the response in an `error` key instead of
  /// `output`. This lets the model distinguish between successful and failed
  /// tool calls.
  /// [imageData] optional image attachments to include in the response.
  /// For Gemini 3+ models, these are nested inside `functionResponse.parts`.
  /// For older models, they are ignored (caller should add them separately).
  /// [modelId] the target model ID, used to determine Gemini 3 behavior.
  static gai.FunctionResponsePart toGeminiFunctionResponse({
    required String functionName,
    required String content,
    bool isError = false,
    List<MediaAttachment>? imageData,
    String? modelId,
  }) {
    // Build the response map — use "output" for success, "error" for errors.
    // For success: try to parse JSON content to preserve structured data.
    // For error: wrap the raw error string.
    late final Map<String, dynamic> response;
    if (isError) {
      response = {'error': content};
    } else {
      Object outputValue;
      try {
        outputValue = jsonDecode(content);
      } catch (_) {
        outputValue = content;
      }
      response = {'output': outputValue};
    }

    // Build multimodal parts for Gemini 3 models.
    List<gai.FunctionResponseInlinePart>? parts;
    if (imageData != null && imageData.isNotEmpty && isGemini3Model(modelId)) {
      parts = imageData
          .where((a) => a.isInline)
          .map(
            (a) => gai.FunctionResponseInlinePart(
              inlineData: gai.FunctionResponseBlob(
                mimeType: a.mimeType,
                data: a.data,
              ),
            ),
          )
          .toList();
      if (parts.isEmpty) parts = null;
    }

    return gai.FunctionResponsePart(
      gai.FunctionResponse(
        name: functionName,
        response: response,
        parts: parts,
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Schema Sanitization
  // ---------------------------------------------------------------------------

  /// Sanitizes a JSON Schema for Gemini compatibility.
  ///
  /// Gemini's API uses a restricted subset of JSON Schema that does not support
  /// `anyOf`, `oneOf`, `allOf`, or `const`. This method flattens these
  /// constructs into Gemini-compatible schemas.
  static Map<String, dynamic> sanitizeSchema(Map<String, dynamic> schema) {
    return _sanitizeSchemaNode(Map<String, dynamic>.from(schema));
  }

  static Map<String, dynamic> _sanitizeSchemaNode(
    Map<String, dynamic> node,
  ) {
    // Remove OpenAI-specific fields.
    node.remove('strict');
    node.remove('\$schema');
    node.remove('additionalProperties');

    // Flatten anyOf/oneOf into a merged object.
    for (final combiner in ['anyOf', 'oneOf']) {
      if (node.containsKey(combiner)) {
        final variants = node[combiner] as List;
        node.remove(combiner);

        // Handle const patterns: [{const: "a"}, {const: "b"}] → enum: ["a", "b"]
        if (variants.every(
          (v) => v is Map<String, dynamic> && v.containsKey('const'),
        )) {
          node['type'] = 'STRING';
          node['enum'] = variants.map((v) => (v as Map<String, dynamic>)['const']).toList();
          continue;
        }

        // Merge all variant properties into a single object schema.
        final mergedProperties = <String, dynamic>{};
        for (final variant in variants) {
          if (variant is Map<String, dynamic>) {
            final props = variant['properties'] as Map<String, dynamic>?;
            if (props != null) {
              mergedProperties.addAll(props);
            }
          }
        }
        if (mergedProperties.isNotEmpty) {
          node['type'] ??= 'OBJECT';
          node['properties'] = (node['properties'] as Map<String, dynamic>? ?? {})..addAll(mergedProperties);
        }
      }
    }

    // Flatten allOf by merging all schemas.
    if (node.containsKey('allOf')) {
      final schemas = node.remove('allOf') as List;
      for (final schema in schemas) {
        if (schema is Map<String, dynamic>) {
          final sanitized = _sanitizeSchemaNode(
            Map<String, dynamic>.from(schema),
          );
          for (final entry in sanitized.entries) {
            if (entry.key == 'properties' && node.containsKey('properties')) {
              (node['properties'] as Map<String, dynamic>).addAll(
                entry.value as Map<String, dynamic>,
              );
            } else if (entry.key == 'required' && node.containsKey('required')) {
              (node['required'] as List).addAll(entry.value as List);
            } else {
              node[entry.key] = entry.value;
            }
          }
        }
      }
    }

    // Remove unsupported const field.
    if (node.containsKey('const')) {
      final constVal = node.remove('const');
      node['type'] = 'STRING';
      node['enum'] = [constVal];
    }

    // Recursively sanitize nested schemas.
    if (node['properties'] is Map) {
      final props = Map<String, dynamic>.from(node['properties'] as Map);
      node['properties'] = props;
      for (final key in props.keys.toList()) {
        if (props[key] is Map) {
          props[key] = _sanitizeSchemaNode(
            Map<String, dynamic>.from(props[key] as Map),
          );
        }
      }
    }
    if (node['items'] is Map) {
      node['items'] = _sanitizeSchemaNode(
        Map<String, dynamic>.from(node['items'] as Map),
      );
    }

    return node;
  }
}
