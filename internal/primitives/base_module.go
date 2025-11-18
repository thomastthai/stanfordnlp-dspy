package primitives

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
)

// BaseModule provides common functionality for all DSPy modules.
// It implements parameter tracking and serialization.
type BaseModule struct {
	// compiled indicates if the module has been compiled/optimized
	compiled bool

	// mu protects concurrent access to the module
	mu sync.RWMutex
}

// NewBaseModule creates a new BaseModule instance.
func NewBaseModule() *BaseModule {
	return &BaseModule{
		compiled: false,
	}
}

// NamedParameters collects all parameters from this module and its submodules.
// Unlike PyTorch, this also handles non-recursive lists of parameters.
func (m *BaseModule) NamedParameters() []NamedParameter {
	m.mu.RLock()
	defer m.mu.RUnlock()

	visited := make(map[uintptr]bool)
	var parameters []NamedParameter

	// Use reflection to find all Parameter fields
	m.collectParameters("", reflect.ValueOf(m).Elem(), visited, &parameters)

	return parameters
}

// collectParameters recursively collects parameters from a value.
func (m *BaseModule) collectParameters(prefix string, v reflect.Value, visited map[uintptr]bool, params *[]NamedParameter) {
	if !v.IsValid() {
		return
	}

	// Handle pointers
	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return
		}

		// Check if we've already visited this pointer
		ptr := v.Pointer()
		if visited[ptr] {
			return
		}
		visited[ptr] = true

		v = v.Elem()
	}

	// Check if this is a Parameter
	if param, ok := v.Interface().(*Parameter); ok {
		name := prefix
		if name == "" {
			name = "self"
		}
		*params = append(*params, NamedParameter{
			Name:  name,
			Param: param,
		})
		return
	}

	// Check if this is a Module
	if mod, ok := v.Interface().(Module); ok {
		// Collect parameters from submodule
		subParams := mod.NamedParameters()
		for _, sp := range subParams {
			name := sp.Name
			if prefix != "" {
				name = prefix + "." + sp.Name
			}
			*params = append(*params, NamedParameter{
				Name:  name,
				Param: sp.Param,
			})
		}
		return
	}

	// Handle structs
	if v.Kind() == reflect.Struct {
		t := v.Type()
		for i := 0; i < v.NumField(); i++ {
			field := v.Field(i)
			fieldType := t.Field(i)

			// Skip unexported fields
			if !field.CanInterface() {
				continue
			}

			fieldName := fieldType.Name
			if prefix != "" {
				fieldName = prefix + "." + fieldName
			}

			m.collectParameters(fieldName, field, visited, params)
		}
		return
	}

	// Handle slices and arrays
	if v.Kind() == reflect.Slice || v.Kind() == reflect.Array {
		for i := 0; i < v.Len(); i++ {
			indexName := fmt.Sprintf("%s[%d]", prefix, i)
			m.collectParameters(indexName, v.Index(i), visited, params)
		}
		return
	}

	// Handle maps
	if v.Kind() == reflect.Map {
		for _, key := range v.MapKeys() {
			keyName := fmt.Sprintf("%s[%v]", prefix, key.Interface())
			m.collectParameters(keyName, v.MapIndex(key), visited, params)
		}
		return
	}
}

// Reset resets all parameters to their initial state.
func (m *BaseModule) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, np := range m.NamedParameters() {
		np.Param.Reset()
	}
}

// SetCompiled marks the module as compiled/optimized, which freezes its parameters.
func (m *BaseModule) SetCompiled(compiled bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.compiled = compiled
}

// IsCompiled returns whether the module is compiled.
func (m *BaseModule) IsCompiled() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.compiled
}

// Save serializes the module to JSON format.
func (m *BaseModule) Save() ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Collect all parameters
	params := m.NamedParameters()

	// Create a map of parameter names to values
	paramMap := make(map[string]interface{})
	for _, np := range params {
		paramMap[np.Name] = np.Param.Value()
	}

	// Serialize to JSON
	return json.Marshal(map[string]interface{}{
		"parameters": paramMap,
		"compiled":   m.compiled,
	})
}

// Load deserializes the module from JSON format.
func (m *BaseModule) Load(data []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var state map[string]interface{}
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("failed to unmarshal module state: %w", err)
	}

	// Load parameters
	paramMap, ok := state["parameters"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid parameters in state")
	}

	params := m.NamedParameters()
	for _, np := range params {
		if value, exists := paramMap[np.Name]; exists {
			np.Param.SetValue(value)
		}
	}

	// Load compiled flag
	if compiled, ok := state["compiled"].(bool); ok {
		m.compiled = compiled
	}

	return nil
}
