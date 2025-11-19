package anthropic

// Tool represents a function/tool definition for Claude.
type Tool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

// ToolUse represents a tool use block in a message.
type ToolUse struct {
	Type  string                 `json:"type"` // "tool_use"
	ID    string                 `json:"id"`
	Name  string                 `json:"name"`
	Input map[string]interface{} `json:"input"`
}

// ToolResult represents a tool result block in a message.
type ToolResult struct {
	Type      string `json:"type"` // "tool_result"
	ToolUseID string `json:"tool_use_id"`
	Content   string `json:"content"`
	IsError   bool   `json:"is_error,omitempty"`
}

// NewTool creates a new tool definition.
func NewTool(name, description string, inputSchema map[string]interface{}) Tool {
	return Tool{
		Name:        name,
		Description: description,
		InputSchema: inputSchema,
	}
}

// NewJSONSchemaForTool creates a JSON schema for a tool's input parameters.
// This is a helper to build the input_schema field.
func NewJSONSchemaForTool(properties map[string]interface{}, required []string) map[string]interface{} {
	schema := map[string]interface{}{
		"type":       "object",
		"properties": properties,
	}

	if len(required) > 0 {
		schema["required"] = required
	}

	return schema
}

// Example tool definitions for common use cases

// GetWeatherTool returns a sample weather tool definition.
func GetWeatherTool() Tool {
	return NewTool(
		"get_weather",
		"Get the current weather in a given location",
		NewJSONSchemaForTool(
			map[string]interface{}{
				"location": map[string]interface{}{
					"type":        "string",
					"description": "The city and state, e.g. San Francisco, CA",
				},
				"unit": map[string]interface{}{
					"type":        "string",
					"enum":        []string{"celsius", "fahrenheit"},
					"description": "The unit of temperature",
				},
			},
			[]string{"location"},
		),
	)
}

// CalculatorTool returns a sample calculator tool definition.
func CalculatorTool() Tool {
	return NewTool(
		"calculator",
		"Perform mathematical calculations",
		NewJSONSchemaForTool(
			map[string]interface{}{
				"expression": map[string]interface{}{
					"type":        "string",
					"description": "The mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')",
				},
			},
			[]string{"expression"},
		),
	)
}

// SearchTool returns a sample search tool definition.
func SearchTool() Tool {
	return NewTool(
		"search",
		"Search for information on the internet",
		NewJSONSchemaForTool(
			map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "The search query",
				},
				"max_results": map[string]interface{}{
					"type":        "integer",
					"description": "Maximum number of results to return",
					"default":     5,
				},
			},
			[]string{"query"},
		),
	)
}
