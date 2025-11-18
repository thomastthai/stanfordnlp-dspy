package primitives

import (
	"encoding/json"
	"sync"
)

// Parameter represents a learnable parameter in a DSPy module.
// This is analogous to demonstrations or prompts that can be optimized.
type Parameter struct {
	value interface{}
	mu    sync.RWMutex
}

// NewParameter creates a new Parameter with the given initial value.
func NewParameter(value interface{}) *Parameter {
	return &Parameter{
		value: value,
	}
}

// Value returns the current value of the parameter.
func (p *Parameter) Value() interface{} {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.value
}

// SetValue updates the parameter's value.
func (p *Parameter) SetValue(value interface{}) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.value = value
}

// Reset resets the parameter to nil.
func (p *Parameter) Reset() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.value = nil
}

// MarshalJSON implements json.Marshaler.
func (p *Parameter) MarshalJSON() ([]byte, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return json.Marshal(p.value)
}

// UnmarshalJSON implements json.Unmarshaler.
func (p *Parameter) UnmarshalJSON(data []byte) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	return json.Unmarshal(data, &p.value)
}
