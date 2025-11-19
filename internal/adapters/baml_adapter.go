package adapters

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

// BAMLAdapter extends JSONAdapter to provide improved structured output rendering.
// It uses a BAML-inspired format that creates human-readable, token-efficient schema
// representations for complex nested structures.
//
// BAML (Behavioral Adaptation Markup Language) style formatting helps LMs better
// understand and generate structured outputs by providing clear schema with comments.
type BAMLAdapter struct {
	*JSONAdapter
	commentSymbol string
}

// NewBAMLAdapter creates a new BAML adapter.
func NewBAMLAdapter() *BAMLAdapter {
	return &BAMLAdapter{
		JSONAdapter:   NewJSONAdapter(),
		commentSymbol: "#",
	}
}

// NewBAMLAdapterWithCommentSymbol creates a BAML adapter with a custom comment symbol.
func NewBAMLAdapterWithCommentSymbol(commentSymbol string) *BAMLAdapter {
	return &BAMLAdapter{
		JSONAdapter:   NewJSONAdapter(),
		commentSymbol: commentSymbol,
	}
}

// Format implements Adapter.Format with BAML-style formatting.
func (a *BAMLAdapter) Format(sig *signatures.Signature, inputs map[string]interface{}, demos []map[string]interface{}) (*clients.Request, error) {
	request := clients.NewRequest()

	// Build system message with BAML-style instructions
	systemMsg := a.buildBAMLSystemMessage(sig)
	request.WithMessages(clients.NewMessage("system", systemMsg))

	// Add demonstrations with BAML formatting
	for _, demo := range demos {
		// Format demo input
		userMsg := a.formatBAMLUserMessage(sig, demo, false)
		request.WithMessages(clients.NewMessage("user", userMsg))

		// Format demo output
		assistantMsg := a.formatBAMLOutputJSON(sig, demo)
		request.WithMessages(clients.NewMessage("assistant", assistantMsg))
	}

	// Add current input with BAML formatting
	userMsg := a.formatBAMLUserMessage(sig, inputs, true)
	request.WithMessages(clients.NewMessage("user", userMsg))

	// Set response format to JSON
	request.Config["response_format"] = map[string]string{"type": "json_object"}

	return request, nil
}

// buildBAMLSystemMessage creates the system message with BAML-style formatting.
func (a *BAMLAdapter) buildBAMLSystemMessage(sig *signatures.Signature) string {
	var sb strings.Builder

	if sig.Instructions != "" {
		sb.WriteString(sig.Instructions)
		sb.WriteString("\n\n")
	}

	sb.WriteString("You must respond with valid JSON.\n\n")

	// Add field descriptions
	sb.WriteString(a.formatFieldDescriptions(sig))
	sb.WriteString("\n\n")

	// Add field structure with BAML-style schema
	sb.WriteString(a.formatFieldStructure(sig))

	return sb.String()
}

// formatFieldDescriptions formats field descriptions in BAML style.
func (a *BAMLAdapter) formatFieldDescriptions(sig *signatures.Signature) string {
	var sb strings.Builder

	// Input fields
	if len(sig.InputFields) > 0 {
		sb.WriteString("Your input fields are:\n")
		for i, field := range sig.InputFields {
			typeName := a.getTypeName(field)
			sb.WriteString(fmt.Sprintf("%d. `%s` (%s)", i+1, field.Name, typeName))
			if field.Description != "" {
				sb.WriteString(fmt.Sprintf(": %s", field.Description))
			}
			sb.WriteString("\n")
		}
	}

	// Output fields
	if len(sig.OutputFields) > 0 {
		sb.WriteString("\nYour output fields are:\n")
		for i, field := range sig.OutputFields {
			typeName := a.getTypeName(field)
			sb.WriteString(fmt.Sprintf("%d. `%s` (%s)", i+1, field.Name, typeName))
			if field.Description != "" {
				sb.WriteString(fmt.Sprintf(": %s", field.Description))
			}
			sb.WriteString("\n")
		}
	}

	return sb.String()
}

// formatFieldStructure formats the field structure in BAML style.
func (a *BAMLAdapter) formatFieldStructure(sig *signatures.Signature) string {
	var sb strings.Builder

	sb.WriteString("All interactions will be structured in the following way, with the appropriate values filled in.\n\n")

	// Input structure
	if len(sig.InputFields) > 0 {
		for _, field := range sig.InputFields {
			sb.WriteString(fmt.Sprintf("[[ ## %s ## ]]\n", field.Name))
			sb.WriteString(fmt.Sprintf("{%s}\n\n", field.Name))
		}
	}

	// Output structure with type information
	if len(sig.OutputFields) > 0 {
		for _, field := range sig.OutputFields {
			sb.WriteString(fmt.Sprintf("[[ ## %s ## ]]\n", field.Name))
			typeStr := a.renderTypeString(field)
			sb.WriteString(fmt.Sprintf("Output field `%s` should be of type: %s\n\n", field.Name, typeStr))
		}
	}

	sb.WriteString("[[ ## completed ## ]]")

	return sb.String()
}

// formatBAMLUserMessage formats the user message with BAML structure.
func (a *BAMLAdapter) formatBAMLUserMessage(sig *signatures.Signature, inputs map[string]interface{}, isMainRequest bool) string {
	var sb strings.Builder

	for _, field := range sig.InputFields {
		if val, ok := inputs[field.Name]; ok {
			sb.WriteString(fmt.Sprintf("[[ ## %s ## ]]\n", field.Name))

			// Format value - use JSON for complex types
			formattedValue := a.formatFieldValue(val)
			sb.WriteString(formattedValue)
			sb.WriteString("\n\n")
		}
	}

	if isMainRequest {
		outputReqs := a.formatOutputRequirements(sig)
		if outputReqs != "" {
			sb.WriteString(outputReqs)
		}
	}

	return strings.TrimSpace(sb.String())
}

// formatBAMLOutputJSON formats the output as JSON.
func (a *BAMLAdapter) formatBAMLOutputJSON(sig *signatures.Signature, outputs map[string]interface{}) string {
	outputData := make(map[string]interface{})
	for _, field := range sig.OutputFields {
		if val, ok := outputs[field.Name]; ok {
			outputData[field.Name] = val
		}
	}

	jsonBytes, err := json.MarshalIndent(outputData, "", "  ")
	if err != nil {
		return fmt.Sprintf("%v", outputData)
	}

	return string(jsonBytes)
}

// formatOutputRequirements formats output requirements for the main request.
func (a *BAMLAdapter) formatOutputRequirements(sig *signatures.Signature) string {
	if len(sig.OutputFields) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("Respond with JSON containing the following fields:\n")

	for _, field := range sig.OutputFields {
		typeStr := a.renderTypeString(field)
		sb.WriteString(fmt.Sprintf("- `%s`: %s", field.Name, typeStr))
		if field.Description != "" {
			sb.WriteString(fmt.Sprintf(" (%s)", field.Description))
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// renderTypeString renders a type string in BAML style.
func (a *BAMLAdapter) renderTypeString(field *signatures.Field) string {
	if field.Type == "" {
		return "string"
	}

	// Map common Go types to BAML-style types
	switch field.Type {
	case "string", "str":
		return "string"
	case "int", "int64", "int32":
		return "int"
	case "float", "float64", "float32":
		return "float"
	case "bool", "boolean":
		return "boolean"
	case "[]string", "list[string]", "array[string]":
		return "string[]"
	case "[]int", "list[int]", "array[int]":
		return "int[]"
	case "map", "dict", "object":
		return "object"
	default:
		// For complex types, return as-is
		return field.Type
	}
}

// getTypeName gets the type name for a field.
func (a *BAMLAdapter) getTypeName(field *signatures.Field) string {
	if field.Type != "" {
		return field.Type
	}
	return "string"
}

// formatFieldValue formats a field value for display.
func (a *BAMLAdapter) formatFieldValue(val interface{}) string {
	if val == nil {
		return "null"
	}

	// Check if it's a complex type that should be JSON formatted
	valType := reflect.TypeOf(val)
	valKind := valType.Kind()

	switch valKind {
	case reflect.Map, reflect.Slice, reflect.Struct, reflect.Ptr:
		// Use JSON formatting for complex types
		jsonBytes, err := json.MarshalIndent(val, "", "  ")
		if err != nil {
			return fmt.Sprintf("%v", val)
		}
		return string(jsonBytes)
	default:
		// Simple types as strings
		return fmt.Sprintf("%v", val)
	}
}

// buildSimplifiedSchema builds a simplified schema representation.
// This is inspired by BAML's schema format for nested structures.
func (a *BAMLAdapter) buildSimplifiedSchema(typeName string, indent int) string {
	currentIndent := strings.Repeat("  ", indent)
	nextIndent := strings.Repeat("  ", indent+1)

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("%s{\n", currentIndent))
	sb.WriteString(fmt.Sprintf("%s%s Note: Define fields here\n", nextIndent, a.commentSymbol))
	sb.WriteString(fmt.Sprintf("%s}\n", currentIndent))

	return sb.String()
}
