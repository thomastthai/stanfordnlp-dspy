package utils

import (
	"context"
	"testing"
)

func TestDummyLM(t *testing.T) {
	responses := []string{"Paris", "London"}
	lm := NewDummyLM(responses)

	// First call
	resp1 := lm.GetResponse("What is the capital of France?")
	if resp1 != "Paris" {
		t.Errorf("Expected response 'Paris', got '%v'", resp1)
	}

	// Second call
	resp2 := lm.GetResponse("What is the capital of England?")
	if resp2 != "London" {
		t.Errorf("Expected response 'London', got '%v'", resp2)
	}

	// Reset and verify
	lm.Reset()
	resp3 := lm.GetResponse("test")
	if resp3 != "Paris" {
		t.Errorf("Expected 'Paris' after reset, got '%v'", resp3)
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
