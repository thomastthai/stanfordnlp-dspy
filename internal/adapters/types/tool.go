package types

import (
	"encoding/json"
	"fmt"
)

// Tool represents a tool/function that can be called by the LM.
type Tool struct {
	// Type is the tool type (usually "function")
	Type string `json:"type"`

	// Function contains the function definition
	Function FunctionDefinition `json:"function"`
}

// FunctionDefinition defines a callable function.
type FunctionDefinition struct {
	// Name is the function name
	Name string `json:"name"`

	// Description describes what the function does
	Description string `json:"description,omitempty"`

	// Parameters is the JSON schema for the function parameters
	Parameters *JSONSchema `json:"parameters,omitempty"`
}

// JSONSchema represents a JSON schema for function parameters.
type JSONSchema struct {
	// Type is the schema type (e.g., "object")
	Type string `json:"type"`

	// Properties defines the schema properties
	Properties map[string]Property `json:"properties,omitempty"`

	// Required lists required property names
	Required []string `json:"required,omitempty"`

	// AdditionalProperties indicates if additional properties are allowed
	AdditionalProperties bool `json:"additionalProperties,omitempty"`
}

// Property represents a schema property.
type Property struct {
	// Type is the property type (e.g., "string", "number", "boolean", "array", "object")
	Type string `json:"type"`

	// Description describes the property
	Description string `json:"description,omitempty"`

	// Enum lists allowed values (for enum types)
	Enum []interface{} `json:"enum,omitempty"`

	// Items defines array item schema (for array types)
	Items *Property `json:"items,omitempty"`

	// Properties defines object properties (for object types)
	Properties map[string]Property `json:"properties,omitempty"`

	// Required lists required property names (for object types)
	Required []string `json:"required,omitempty"`
}

// NewTool creates a new tool with a function definition.
func NewTool(name, description string, parameters *JSONSchema) *Tool {
	return &Tool{
		Type: "function",
		Function: FunctionDefinition{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		},
	}
}

// NewJSONSchema creates a new JSON schema for object parameters.
func NewJSONSchema() *JSONSchema {
	return &JSONSchema{
		Type:                 "object",
		Properties:           make(map[string]Property),
		Required:             []string{},
		AdditionalProperties: false,
	}
}

// AddProperty adds a property to the schema.
func (s *JSONSchema) AddProperty(name string, prop Property) *JSONSchema {
	s.Properties[name] = prop
	return s
}

// AddRequiredProperty adds a required property to the schema.
func (s *JSONSchema) AddRequiredProperty(name string, prop Property) *JSONSchema {
	s.Properties[name] = prop
	s.Required = append(s.Required, name)
	return s
}

// NewStringProperty creates a string property.
func NewStringProperty(description string) Property {
	return Property{
		Type:        "string",
		Description: description,
	}
}

// NewNumberProperty creates a number property.
func NewNumberProperty(description string) Property {
	return Property{
		Type:        "number",
		Description: description,
	}
}

// NewBooleanProperty creates a boolean property.
func NewBooleanProperty(description string) Property {
	return Property{
		Type:        "boolean",
		Description: description,
	}
}

// NewArrayProperty creates an array property.
func NewArrayProperty(description string, items Property) Property {
	return Property{
		Type:        "array",
		Description: description,
		Items:       &items,
	}
}

// NewEnumProperty creates an enum property.
func NewEnumProperty(description string, values []interface{}) Property {
	return Property{
		Type:        "string",
		Description: description,
		Enum:        values,
	}
}

// Validate validates a tool definition.
func (t *Tool) Validate() error {
	if t.Type == "" {
		return fmt.Errorf("tool type is required")
	}

	if t.Function.Name == "" {
		return fmt.Errorf("function name is required")
	}

	// Validate parameters schema if present
	if t.Function.Parameters != nil {
		if err := t.Function.Parameters.Validate(); err != nil {
			return fmt.Errorf("invalid parameters schema: %w", err)
		}
	}

	return nil
}

// Validate validates a JSON schema.
func (s *JSONSchema) Validate() error {
	if s.Type == "" {
		return fmt.Errorf("schema type is required")
	}

	// Validate that required properties exist
	for _, req := range s.Required {
		if _, ok := s.Properties[req]; !ok {
			return fmt.Errorf("required property '%s' not defined in schema", req)
		}
	}

	return nil
}

// ToJSON serializes the tool to JSON.
func (t *Tool) ToJSON() (string, error) {
	data, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal tool: %w", err)
	}
	return string(data), nil
}

// FromJSON deserializes a tool from JSON.
func FromJSON(data string) (*Tool, error) {
	var tool Tool
	if err := json.Unmarshal([]byte(data), &tool); err != nil {
		return nil, fmt.Errorf("failed to unmarshal tool: %w", err)
	}
	return &tool, nil
}
