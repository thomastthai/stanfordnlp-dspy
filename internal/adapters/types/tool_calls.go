package types

import (
	"encoding/json"
	"fmt"
)

// ToolCall represents a single tool/function call request from the LM.
type ToolCall struct {
	// ID is the unique identifier for this tool call
	ID string `json:"id"`

	// Type is the call type (usually "function")
	Type string `json:"type"`

	// Function contains the function call details
	Function FunctionCall `json:"function"`
}

// FunctionCall represents a function call.
type FunctionCall struct {
	// Name is the function name
	Name string `json:"name"`

	// Arguments is the JSON-encoded function arguments
	Arguments string `json:"arguments"`
}

// ToolCallResult represents the result of executing a tool call.
type ToolCallResult struct {
	// ToolCallID is the ID of the tool call this result is for
	ToolCallID string `json:"tool_call_id"`

	// Content is the result content
	Content string `json:"content"`

	// Error is any error that occurred during execution
	Error error `json:"-"`
}

// ToolCallOrchestrator manages multiple tool calls and their execution.
type ToolCallOrchestrator struct {
	// ToolCalls is the list of tool calls to execute
	ToolCalls []ToolCall

	// Results stores the execution results
	Results []ToolCallResult

	// ToolRegistry maps tool names to their implementations
	ToolRegistry map[string]ToolImplementation
}

// ToolImplementation is a function that implements a tool.
type ToolImplementation func(arguments map[string]interface{}) (interface{}, error)

// NewToolCallOrchestrator creates a new orchestrator.
func NewToolCallOrchestrator() *ToolCallOrchestrator {
	return &ToolCallOrchestrator{
		ToolCalls:    []ToolCall{},
		Results:      []ToolCallResult{},
		ToolRegistry: make(map[string]ToolImplementation),
	}
}

// RegisterTool registers a tool implementation.
func (o *ToolCallOrchestrator) RegisterTool(name string, impl ToolImplementation) {
	o.ToolRegistry[name] = impl
}

// AddToolCall adds a tool call to the orchestrator.
func (o *ToolCallOrchestrator) AddToolCall(call ToolCall) {
	o.ToolCalls = append(o.ToolCalls, call)
}

// ExecuteAll executes all registered tool calls.
func (o *ToolCallOrchestrator) ExecuteAll() error {
	o.Results = make([]ToolCallResult, 0, len(o.ToolCalls))

	for _, call := range o.ToolCalls {
		result, err := o.executeToolCall(call)
		if err != nil {
			result.Error = err
		}
		o.Results = append(o.Results, result)
	}

	return nil
}

// executeToolCall executes a single tool call.
func (o *ToolCallOrchestrator) executeToolCall(call ToolCall) (ToolCallResult, error) {
	result := ToolCallResult{
		ToolCallID: call.ID,
	}

	// Get tool implementation
	impl, ok := o.ToolRegistry[call.Function.Name]
	if !ok {
		result.Error = fmt.Errorf("tool not found: %s", call.Function.Name)
		result.Content = fmt.Sprintf("Error: tool '%s' not found", call.Function.Name)
		return result, result.Error
	}

	// Parse arguments
	var args map[string]interface{}
	if call.Function.Arguments != "" {
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			result.Error = fmt.Errorf("failed to parse arguments: %w", err)
			result.Content = fmt.Sprintf("Error: invalid arguments for tool '%s'", call.Function.Name)
			return result, result.Error
		}
	} else {
		args = make(map[string]interface{})
	}

	// Execute tool
	output, err := impl(args)
	if err != nil {
		result.Error = err
		result.Content = fmt.Sprintf("Error executing tool '%s': %v", call.Function.Name, err)
		return result, err
	}

	// Convert output to string
	switch v := output.(type) {
	case string:
		result.Content = v
	case []byte:
		result.Content = string(v)
	default:
		// Serialize to JSON
		data, err := json.Marshal(output)
		if err != nil {
			result.Error = fmt.Errorf("failed to serialize output: %w", err)
			result.Content = fmt.Sprintf("Error: %v", output)
			return result, err
		}
		result.Content = string(data)
	}

	return result, nil
}

// GetResults returns all execution results.
func (o *ToolCallOrchestrator) GetResults() []ToolCallResult {
	return o.Results
}

// HasErrors returns true if any tool call resulted in an error.
func (o *ToolCallOrchestrator) HasErrors() bool {
	for _, result := range o.Results {
		if result.Error != nil {
			return true
		}
	}
	return false
}

// ParseToolCall parses a tool call from JSON.
func ParseToolCall(data string) (*ToolCall, error) {
	var call ToolCall
	if err := json.Unmarshal([]byte(data), &call); err != nil {
		return nil, fmt.Errorf("failed to parse tool call: %w", err)
	}
	return &call, nil
}

// ParseToolCalls parses multiple tool calls from JSON.
func ParseToolCalls(data string) ([]ToolCall, error) {
	var calls []ToolCall
	if err := json.Unmarshal([]byte(data), &calls); err != nil {
		return nil, fmt.Errorf("failed to parse tool calls: %w", err)
	}
	return calls, nil
}

// ToJSON serializes a tool call to JSON.
func (tc *ToolCall) ToJSON() (string, error) {
	data, err := json.Marshal(tc)
	if err != nil {
		return "", fmt.Errorf("failed to marshal tool call: %w", err)
	}
	return string(data), nil
}

// ParseArguments parses the function arguments.
func (fc *FunctionCall) ParseArguments() (map[string]interface{}, error) {
	var args map[string]interface{}
	if fc.Arguments == "" {
		return make(map[string]interface{}), nil
	}

	if err := json.Unmarshal([]byte(fc.Arguments), &args); err != nil {
		return nil, fmt.Errorf("failed to parse arguments: %w", err)
	}

	return args, nil
}

// Validate validates a tool call.
func (tc *ToolCall) Validate() error {
	if tc.ID == "" {
		return fmt.Errorf("tool call ID is required")
	}

	if tc.Type == "" {
		tc.Type = "function" // Default to function
	}

	if tc.Function.Name == "" {
		return fmt.Errorf("function name is required")
	}

	// Validate arguments are valid JSON if not empty
	if tc.Function.Arguments != "" {
		var args map[string]interface{}
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			return fmt.Errorf("invalid arguments JSON: %w", err)
		}
	}

	return nil
}

// ToMessageContent converts a tool call result to message content for LM APIs.
func (r *ToolCallResult) ToMessageContent() map[string]interface{} {
	return map[string]interface{}{
		"role":         "tool",
		"tool_call_id": r.ToolCallID,
		"content":      r.Content,
	}
}
