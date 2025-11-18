package primitives

import (
	"encoding/json"
	"testing"
)

func TestNewExample(t *testing.T) {
	inputs := map[string]interface{}{
		"question": "What is Go?",
	}
	outputs := map[string]interface{}{
		"answer": "A programming language",
	}
	
	ex := NewExample(inputs, outputs)
	
	if ex == nil {
		t.Fatal("expected non-nil example")
	}
	
	if len(ex.Inputs()) != 1 {
		t.Errorf("expected 1 input, got %d", len(ex.Inputs()))
	}
	
	if len(ex.Outputs()) != 1 {
		t.Errorf("expected 1 output, got %d", len(ex.Outputs()))
	}
}

func TestExample_Get(t *testing.T) {
	ex := NewExample(
		map[string]interface{}{"input": "test"},
		map[string]interface{}{"output": "result"},
	)
	
	// Get input field
	val, ok := ex.Get("input")
	if !ok {
		t.Error("expected to find 'input' field")
	}
	if val != "test" {
		t.Errorf("expected 'test', got %v", val)
	}
	
	// Get output field
	val, ok = ex.Get("output")
	if !ok {
		t.Error("expected to find 'output' field")
	}
	if val != "result" {
		t.Errorf("expected 'result', got %v", val)
	}
	
	// Get non-existent field
	_, ok = ex.Get("missing")
	if ok {
		t.Error("expected not to find 'missing' field")
	}
}

func TestExample_Set(t *testing.T) {
	ex := NewExample(
		map[string]interface{}{"input": "test"},
		map[string]interface{}{},
	)
	
	// Set existing input field
	ex.Set("input", "updated")
	val, _ := ex.Get("input")
	if val != "updated" {
		t.Errorf("expected 'updated', got %v", val)
	}
	
	// Set new field (should go to outputs)
	ex.Set("new", "value")
	val, _ = ex.Get("new")
	if val != "value" {
		t.Errorf("expected 'value', got %v", val)
	}
}

func TestExample_With(t *testing.T) {
	ex := NewExample(
		map[string]interface{}{"input": "test"},
		map[string]interface{}{"output": "result"},
	)
	
	ex2 := ex.With(map[string]interface{}{
		"new_field": "new_value",
	})
	
	// Original should be unchanged
	if _, ok := ex.Get("new_field"); ok {
		t.Error("original example should not be modified")
	}
	
	// New example should have all fields
	if _, ok := ex2.Get("input"); !ok {
		t.Error("new example should have 'input'")
	}
	if _, ok := ex2.Get("output"); !ok {
		t.Error("new example should have 'output'")
	}
	if _, ok := ex2.Get("new_field"); !ok {
		t.Error("new example should have 'new_field'")
	}
}

func TestExample_Metadata(t *testing.T) {
	ex := NewExample(nil, nil)
	
	ex.SetMetadata("key", "value")
	
	val, ok := ex.GetMetadata("key")
	if !ok {
		t.Error("expected to find metadata")
	}
	if val != "value" {
		t.Errorf("expected 'value', got %v", val)
	}
}

func TestExample_JSON(t *testing.T) {
	ex := NewExample(
		map[string]interface{}{"input": "test"},
		map[string]interface{}{"output": "result"},
	)
	ex.SetMetadata("meta", "data")
	
	// Marshal
	data, err := json.Marshal(ex)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}
	
	// Unmarshal
	var ex2 Example
	if err := json.Unmarshal(data, &ex2); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}
	
	// Verify fields
	if val, _ := ex2.Get("input"); val != "test" {
		t.Errorf("expected 'test', got %v", val)
	}
	if val, _ := ex2.Get("output"); val != "result" {
		t.Errorf("expected 'result', got %v", val)
	}
	if val, _ := ex2.GetMetadata("meta"); val != "data" {
		t.Errorf("expected 'data', got %v", val)
	}
}
