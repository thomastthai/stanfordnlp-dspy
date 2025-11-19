package utils

import (
	"context"
	"testing"
)

func TestDummyLM(t *testing.T) {
	responses := []map[string]interface{}{
		{"answer": "Paris"},
		{"answer": "London"},
	}

	lm := NewDummyLM(DummyLMOptions{
		Responses: responses,
	})

	ctx := context.Background()

	// First call
	resp1, err := lm.Call(ctx, []map[string]interface{}{
		{"role": "user", "content": "What is the capital of France?"},
	})
	if err != nil {
		t.Fatalf("Call() error = %v", err)
	}
	if resp1["answer"] != "Paris" {
		t.Errorf("Expected answer 'Paris', got '%v'", resp1["answer"])
	}

	// Second call
	resp2, err := lm.Call(ctx, []map[string]interface{}{
		{"role": "user", "content": "What is the capital of England?"},
	})
	if err != nil {
		t.Fatalf("Call() error = %v", err)
	}
	if resp2["answer"] != "London" {
		t.Errorf("Expected answer 'London', got '%v'", resp2["answer"])
	}

	// Check history
	history := lm.GetHistory()
	if len(history) != 2 {
		t.Errorf("Expected 2 history entries, got %d", len(history))
	}

	// Reset and verify
	lm.Reset()
	history = lm.GetHistory()
	if len(history) != 0 {
		t.Errorf("Expected 0 history entries after reset, got %d", len(history))
	}
}

func TestDummyRM(t *testing.T) {
	passages := []string{
		"The Eiffel Tower is in Paris",
		"London is the capital of England",
		"Paris is the capital of France",
	}

	rm := NewDummyRM(passages)
	ctx := context.Background()

	// Retrieve with query about Paris
	results, err := rm.Retrieve(ctx, "Paris capital France", 2)
	if err != nil {
		t.Fatalf("Retrieve() error = %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	// First result should contain "Paris" and "France" (highest overlap)
	if results[0] != "Paris is the capital of France" {
		t.Errorf("Expected first result to be about Paris, got '%s'", results[0])
	}

	// Test error handling
	_, err = rm.Retrieve(ctx, "test", 0)
	if err == nil {
		t.Error("Expected error for k=0, got nil")
	}
}
