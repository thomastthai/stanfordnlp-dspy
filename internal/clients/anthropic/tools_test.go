package anthropic

import (
	"testing"
)

func TestNewTool(t *testing.T) {
	tool := NewTool(
		"test_tool",
		"A test tool",
		map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"param1": map[string]interface{}{
					"type": "string",
				},
			},
		},
	)

	if tool.Name != "test_tool" {
		t.Errorf("got name %q, want %q", tool.Name, "test_tool")
	}

	if tool.Description != "A test tool" {
		t.Errorf("got description %q, want %q", tool.Description, "A test tool")
	}

	if tool.InputSchema == nil {
		t.Error("input schema should not be nil")
	}
}

func TestNewJSONSchemaForTool(t *testing.T) {
	properties := map[string]interface{}{
		"location": map[string]interface{}{
			"type":        "string",
			"description": "The location",
		},
	}
	required := []string{"location"}

	schema := NewJSONSchemaForTool(properties, required)

	if schema["type"] != "object" {
		t.Errorf("expected type to be object, got %v", schema["type"])
	}

	if schema["properties"] == nil {
		t.Error("properties should not be nil")
	}

	requiredField, ok := schema["required"].([]string)
	if !ok {
		t.Error("required field should be []string")
	}

	if len(requiredField) != 1 || requiredField[0] != "location" {
		t.Errorf("got required %v, want [location]", requiredField)
	}
}

func TestGetWeatherTool(t *testing.T) {
	tool := GetWeatherTool()

	if tool.Name != "get_weather" {
		t.Errorf("got name %q, want %q", tool.Name, "get_weather")
	}

	if tool.InputSchema == nil {
		t.Error("input schema should not be nil")
	}

	// Check that properties exist
	props, ok := tool.InputSchema["properties"].(map[string]interface{})
	if !ok {
		t.Fatal("properties should be a map")
	}

	if _, ok := props["location"]; !ok {
		t.Error("location property should exist")
	}

	if _, ok := props["unit"]; !ok {
		t.Error("unit property should exist")
	}
}

func TestCalculatorTool(t *testing.T) {
	tool := CalculatorTool()

	if tool.Name != "calculator" {
		t.Errorf("got name %q, want %q", tool.Name, "calculator")
	}

	if tool.InputSchema == nil {
		t.Error("input schema should not be nil")
	}

	props, ok := tool.InputSchema["properties"].(map[string]interface{})
	if !ok {
		t.Fatal("properties should be a map")
	}

	if _, ok := props["expression"]; !ok {
		t.Error("expression property should exist")
	}
}

func TestSearchTool(t *testing.T) {
	tool := SearchTool()

	if tool.Name != "search" {
		t.Errorf("got name %q, want %q", tool.Name, "search")
	}

	if tool.InputSchema == nil {
		t.Error("input schema should not be nil")
	}

	props, ok := tool.InputSchema["properties"].(map[string]interface{})
	if !ok {
		t.Fatal("properties should be a map")
	}

	if _, ok := props["query"]; !ok {
		t.Error("query property should exist")
	}

	if _, ok := props["max_results"]; !ok {
		t.Error("max_results property should exist")
	}
}

func TestToolSchemaStructure(t *testing.T) {
	tests := []struct {
		name string
		tool Tool
	}{
		{"weather", GetWeatherTool()},
		{"calculator", CalculatorTool()},
		{"search", SearchTool()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.tool.Name == "" {
				t.Error("tool name should not be empty")
			}

			if tt.tool.Description == "" {
				t.Error("tool description should not be empty")
			}

			if tt.tool.InputSchema == nil {
				t.Fatal("input schema should not be nil")
			}

			// Check schema structure
			if tt.tool.InputSchema["type"] != "object" {
				t.Error("schema type should be object")
			}

			if tt.tool.InputSchema["properties"] == nil {
				t.Error("schema should have properties")
			}
		})
	}
}
