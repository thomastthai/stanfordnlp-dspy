package primitives

import (
	"encoding/json"
	"fmt"
)

// Prediction represents the output from a DSPy module.
// It contains the predicted fields and optional metadata about the prediction.
type Prediction struct {
	// fields contains the predicted output fields
	fields map[string]interface{}

	// metadata contains additional information (trace, reasoning, etc.)
	metadata map[string]interface{}
}

// NewPrediction creates a new Prediction with the given fields.
func NewPrediction(fields map[string]interface{}) *Prediction {
	if fields == nil {
		fields = make(map[string]interface{})
	}

	return &Prediction{
		fields:   fields,
		metadata: make(map[string]interface{}),
	}
}

// Get returns the value for the given field name.
func (p *Prediction) Get(field string) (interface{}, bool) {
	val, ok := p.fields[field]
	return val, ok
}

// Set sets the value for the given field.
func (p *Prediction) Set(field string, value interface{}) {
	p.fields[field] = value
}

// Fields returns all predicted fields.
func (p *Prediction) Fields() map[string]interface{} {
	return p.fields
}

// SetMetadata sets a metadata field.
func (p *Prediction) SetMetadata(key string, value interface{}) {
	p.metadata[key] = value
}

// GetMetadata returns a metadata field.
func (p *Prediction) GetMetadata(key string) (interface{}, bool) {
	val, ok := p.metadata[key]
	return val, ok
}

// Metadata returns all metadata.
func (p *Prediction) Metadata() map[string]interface{} {
	return p.metadata
}

// ToMap returns a single map with all fields.
func (p *Prediction) ToMap() map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range p.fields {
		result[k] = v
	}
	return result
}

// MarshalJSON implements json.Marshaler.
func (p *Prediction) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]interface{}{
		"fields":   p.fields,
		"metadata": p.metadata,
	})
}

// UnmarshalJSON implements json.Unmarshaler.
func (p *Prediction) UnmarshalJSON(data []byte) error {
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("failed to unmarshal prediction: %w", err)
	}

	if fields, ok := raw["fields"].(map[string]interface{}); ok {
		p.fields = fields
	} else {
		p.fields = make(map[string]interface{})
	}

	if metadata, ok := raw["metadata"].(map[string]interface{}); ok {
		p.metadata = metadata
	} else {
		p.metadata = make(map[string]interface{})
	}

	return nil
}

// String returns a string representation of the prediction.
func (p *Prediction) String() string {
	data, _ := json.MarshalIndent(p, "", "  ")
	return string(data)
}
