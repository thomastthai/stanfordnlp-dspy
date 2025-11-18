package adapters

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

// JSONAdapter formats requests for JSON mode output.
type JSONAdapter struct {
	*BaseAdapter
	strictSchema bool
}

// NewJSONAdapter creates a new JSON adapter.
func NewJSONAdapter() *JSONAdapter {
	return &JSONAdapter{
		BaseAdapter:  NewBaseAdapter("json"),
		strictSchema: false,
	}
}

// NewJSONAdapterWithSchema creates a new JSON adapter with strict schema validation.
func NewJSONAdapterWithSchema() *JSONAdapter {
	return &JSONAdapter{
		BaseAdapter:  NewBaseAdapter("json"),
		strictSchema: true,
	}
}

// Format implements Adapter.Format.
func (a *JSONAdapter) Format(sig *signatures.Signature, inputs map[string]interface{}, demos []map[string]interface{}) (*clients.Request, error) {
	request := clients.NewRequest()

	// Build system message with JSON instructions
	systemMsg := a.buildSystemMessage(sig)
	request.WithMessages(clients.NewMessage("system", systemMsg))

	// Add demonstrations as few-shot examples in JSON format
	for _, demo := range demos {
		// Format demo input
		inputJSON := a.formatInputJSON(sig, demo)
		request.WithMessages(clients.NewMessage("user", inputJSON))

		// Format demo output
		outputJSON := a.formatOutputJSON(sig, demo)
		request.WithMessages(clients.NewMessage("assistant", outputJSON))
	}

	// Add current input in JSON format
	inputJSON := a.formatInputJSON(sig, inputs)
	request.WithMessages(clients.NewMessage("user", inputJSON))

	// Set response format to JSON if supported by the model
	// This would need to be set in the config
	request.Config["response_format"] = map[string]string{"type": "json_object"}

	return request, nil
}

// Parse implements Adapter.Parse.
func (a *JSONAdapter) Parse(sig *signatures.Signature, response *clients.Response) (map[string]interface{}, error) {
	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	content := response.Choices[0].Message.Content

	// Try to parse as JSON
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(content), &result); err != nil {
		// If parsing fails, try to extract JSON from the response
		extracted, extractErr := a.extractJSON(content)
		if extractErr != nil {
			// If extraction also fails, try to repair the JSON
			repaired, repairErr := a.repairJSON(content)
			if repairErr != nil {
				return nil, fmt.Errorf("failed to parse JSON response: %w (original error: %v)", repairErr, err)
			}
			content = repaired
			if err := json.Unmarshal([]byte(content), &result); err != nil {
				return nil, fmt.Errorf("failed to parse repaired JSON: %w", err)
			}
		} else {
			content = extracted
			if err := json.Unmarshal([]byte(content), &result); err != nil {
				return nil, fmt.Errorf("failed to parse extracted JSON: %w", err)
			}
		}
	}

	// Validate schema if strict mode is enabled
	if a.strictSchema {
		if err := a.validateSchema(sig, result); err != nil {
			return nil, fmt.Errorf("schema validation failed: %w", err)
		}
	}

	// Extract only the output fields defined in the signature
	outputs := make(map[string]interface{})
	for _, field := range sig.OutputFields {
		if val, ok := result[field.Name]; ok {
			outputs[field.Name] = val
		} else {
			// Try case-insensitive match
			for k, v := range result {
				if strings.EqualFold(k, field.Name) {
					outputs[field.Name] = v
					break
				}
			}
		}
	}

	// If no output fields matched, return all fields
	if len(outputs) == 0 {
		return result, nil
	}

	return outputs, nil
}

// buildSystemMessage creates the system message with JSON instructions.
func (a *JSONAdapter) buildSystemMessage(sig *signatures.Signature) string {
	var sb strings.Builder

	if sig.Instructions != "" {
		sb.WriteString(sig.Instructions)
		sb.WriteString("\n\n")
	}

	sb.WriteString("You must respond with valid JSON. ")
	
	// Build JSON schema description
	sb.WriteString("The response should be a JSON object with the following fields:\n")
	for _, field := range sig.OutputFields {
		sb.WriteString(fmt.Sprintf("- \"%s\"", field.Name))
		if field.Description != "" {
			sb.WriteString(fmt.Sprintf(": %s", field.Description))
		}
		sb.WriteString("\n")
	}

	if a.strictSchema {
		sb.WriteString("\nOnly include the specified fields in your response.")
	}

	return sb.String()
}

// formatInputJSON formats input fields as JSON.
func (a *JSONAdapter) formatInputJSON(sig *signatures.Signature, data map[string]interface{}) string {
	inputData := make(map[string]interface{})
	for _, field := range sig.InputFields {
		if val, ok := data[field.Name]; ok {
			inputData[field.Name] = val
		}
	}

	jsonBytes, err := json.MarshalIndent(inputData, "", "  ")
	if err != nil {
		// Fallback to simple format
		return fmt.Sprintf("%v", inputData)
	}

	return string(jsonBytes)
}

// formatOutputJSON formats output fields as JSON.
func (a *JSONAdapter) formatOutputJSON(sig *signatures.Signature, data map[string]interface{}) string {
	outputData := make(map[string]interface{})
	for _, field := range sig.OutputFields {
		if val, ok := data[field.Name]; ok {
			outputData[field.Name] = val
		}
	}

	jsonBytes, err := json.MarshalIndent(outputData, "", "  ")
	if err != nil {
		// Fallback to simple format
		return fmt.Sprintf("%v", outputData)
	}

	return string(jsonBytes)
}

// extractJSON tries to extract JSON from text that may contain additional content.
func (a *JSONAdapter) extractJSON(text string) (string, error) {
	// Try to find JSON object boundaries
	start := strings.Index(text, "{")
	if start == -1 {
		return "", fmt.Errorf("no JSON object found")
	}

	// Find matching closing brace
	depth := 0
	for i := start; i < len(text); i++ {
		switch text[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return text[start : i+1], nil
			}
		}
	}

	return "", fmt.Errorf("unmatched braces in JSON")
}

// repairJSON attempts to repair malformed JSON.
func (a *JSONAdapter) repairJSON(text string) (string, error) {
	// Remove markdown code blocks
	text = strings.TrimPrefix(text, "```json")
	text = strings.TrimPrefix(text, "```")
	text = strings.TrimSuffix(text, "```")
	text = strings.TrimSpace(text)

	// Try to extract JSON again after cleanup
	return a.extractJSON(text)
}

// validateSchema validates that the result matches the expected schema.
func (a *JSONAdapter) validateSchema(sig *signatures.Signature, result map[string]interface{}) error {
	// Check that all required output fields are present
	for _, field := range sig.OutputFields {
		found := false
		for k := range result {
			if strings.EqualFold(k, field.Name) {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("required field '%s' not found in response", field.Name)
		}
	}

	return nil
}
