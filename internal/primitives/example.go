package primitives

import (
	"encoding/json"
	"fmt"
)

// Example represents a training example with inputs and outputs.
// Examples are used for few-shot learning and optimization.
type Example struct {
	// inputs contains the input fields
	inputs map[string]interface{}

	// outputs contains the expected output fields (labels)
	outputs map[string]interface{}

	// metadata contains additional information about the example
	metadata map[string]interface{}
}

// NewExample creates a new Example with the given inputs and outputs.
func NewExample(inputs, outputs map[string]interface{}) *Example {
	if inputs == nil {
		inputs = make(map[string]interface{})
	}
	if outputs == nil {
		outputs = make(map[string]interface{})
	}

	return &Example{
		inputs:   inputs,
		outputs:  outputs,
		metadata: make(map[string]interface{}),
	}
}

// Inputs returns the input fields.
func (e *Example) Inputs() map[string]interface{} {
	return e.inputs
}

// Outputs returns the output fields.
func (e *Example) Outputs() map[string]interface{} {
	return e.outputs
}

// Get returns the value for the given field name.
// Looks in both inputs and outputs.
func (e *Example) Get(field string) (interface{}, bool) {
	if val, ok := e.inputs[field]; ok {
		return val, true
	}
	if val, ok := e.outputs[field]; ok {
		return val, true
	}
	return nil, false
}

// Set sets the value for the given field.
// If the field exists in inputs, it updates inputs, otherwise outputs.
func (e *Example) Set(field string, value interface{}) {
	if _, ok := e.inputs[field]; ok {
		e.inputs[field] = value
	} else {
		e.outputs[field] = value
	}
}

// With creates a new Example with additional fields.
func (e *Example) With(fields map[string]interface{}) *Example {
	newExample := &Example{
		inputs:   make(map[string]interface{}),
		outputs:  make(map[string]interface{}),
		metadata: make(map[string]interface{}),
	}

	// Copy existing fields
	for k, v := range e.inputs {
		newExample.inputs[k] = v
	}
	for k, v := range e.outputs {
		newExample.outputs[k] = v
	}
	for k, v := range e.metadata {
		newExample.metadata[k] = v
	}

	// Add new fields
	for k, v := range fields {
		newExample.Set(k, v)
	}

	return newExample
}

// SetMetadata sets a metadata field.
func (e *Example) SetMetadata(key string, value interface{}) {
	e.metadata[key] = value
}

// GetMetadata returns a metadata field.
func (e *Example) GetMetadata(key string) (interface{}, bool) {
	val, ok := e.metadata[key]
	return val, ok
}

// Metadata returns all metadata.
func (e *Example) Metadata() map[string]interface{} {
	return e.metadata
}

// ToMap returns a single map with all fields (inputs and outputs combined).
func (e *Example) ToMap() map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range e.inputs {
		result[k] = v
	}
	for k, v := range e.outputs {
		result[k] = v
	}
	return result
}

// MarshalJSON implements json.Marshaler.
func (e *Example) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]interface{}{
		"inputs":   e.inputs,
		"outputs":  e.outputs,
		"metadata": e.metadata,
	})
}

// UnmarshalJSON implements json.Unmarshaler.
func (e *Example) UnmarshalJSON(data []byte) error {
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("failed to unmarshal example: %w", err)
	}

	if inputs, ok := raw["inputs"].(map[string]interface{}); ok {
		e.inputs = inputs
	} else {
		e.inputs = make(map[string]interface{})
	}

	if outputs, ok := raw["outputs"].(map[string]interface{}); ok {
		e.outputs = outputs
	} else {
		e.outputs = make(map[string]interface{})
	}

	if metadata, ok := raw["metadata"].(map[string]interface{}); ok {
		e.metadata = metadata
	} else {
		e.metadata = make(map[string]interface{})
	}

	return nil
}

// String returns a string representation of the example.
func (e *Example) String() string {
	data, _ := json.MarshalIndent(e, "", "  ")
	return string(data)
}

// WithInputs marks the specified fields as inputs.
// This creates a new Example where the specified fields are moved to inputs.
func (e *Example) WithInputs(fields ...string) *Example {
	newExample := &Example{
		inputs:   make(map[string]interface{}),
		outputs:  make(map[string]interface{}),
		metadata: make(map[string]interface{}),
	}

	// Copy all fields first
	allFields := e.ToMap()
	for k, v := range allFields {
		newExample.outputs[k] = v
	}

	// Move specified fields to inputs
	for _, field := range fields {
		if val, ok := newExample.outputs[field]; ok {
			newExample.inputs[field] = val
			delete(newExample.outputs, field)
		}
	}

	// Copy metadata
	for k, v := range e.metadata {
		newExample.metadata[k] = v
	}

	return newExample
}
