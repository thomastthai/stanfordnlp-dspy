package streaming

import (
	"context"
	"testing"
	"time"
)

func TestStream_Basic(t *testing.T) {
	ctx := context.Background()
	stream := NewStream[int](ctx, 5)

	// Send values
	go func() {
		for i := 0; i < 5; i++ {
			stream.Send(i)
		}
		stream.Close()
	}()

	// Receive values
	var received []int
	for {
		val, ok := stream.Receive()
		if !ok {
			break
		}
		received = append(received, val)
	}

	if len(received) != 5 {
		t.Errorf("Expected 5 values, got %d", len(received))
	}
}

func TestStream_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	stream := NewStream[int](ctx, 0) // Use 0 buffer to ensure blocking

	// Cancel context
	cancel()

	// Give a moment for cancellation to propagate
	time.Sleep(10 * time.Millisecond)

	// Try to send - should fail
	err := stream.Send(42)
	if err == nil {
		t.Error("Expected error when sending to cancelled stream")
	}
}

func TestFilter(t *testing.T) {
	ctx := context.Background()
	input := NewStream[int](ctx, 5)

	// Filter even numbers
	output := Filter(ctx, input, func(n int) bool {
		return n%2 == 0
	})

	// Send values
	go func() {
		for i := 0; i < 10; i++ {
			input.Send(i)
		}
		input.Close()
	}()

	// Collect results
	var results []int
	for val := range output.Chan() {
		results = append(results, val)
	}

	expected := []int{0, 2, 4, 6, 8}
	if len(results) != len(expected) {
		t.Errorf("Expected %d values, got %d", len(expected), len(results))
	}

	for i, val := range results {
		if val != expected[i] {
			t.Errorf("At index %d: expected %d, got %d", i, expected[i], val)
		}
	}
}

func TestMap(t *testing.T) {
	ctx := context.Background()
	input := NewStream[int](ctx, 5)

	// Map: multiply by 2
	output := Map(ctx, input, func(n int) int {
		return n * 2
	})

	// Send values
	go func() {
		for i := 0; i < 5; i++ {
			input.Send(i)
		}
		input.Close()
	}()

	// Collect results
	var results []int
	for val := range output.Chan() {
		results = append(results, val)
	}

	expected := []int{0, 2, 4, 6, 8}
	if len(results) != len(expected) {
		t.Errorf("Expected %d values, got %d", len(expected), len(results))
	}

	for i, val := range results {
		if val != expected[i] {
			t.Errorf("At index %d: expected %d, got %d", i, expected[i], val)
		}
	}
}

func TestAggregate(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	stream := NewStream[string](ctx, 5)

	// Send values in a goroutine
	go func() {
		values := []string{"hello", "world", "test"}
		for _, v := range values {
			stream.Send(v)
		}
		stream.Close()
	}()

	// Aggregate
	results, err := Aggregate(ctx, stream)
	if err != nil {
		t.Fatalf("Aggregate() error = %v", err)
	}

	expected := []string{"hello", "world", "test"}
	if len(results) != len(expected) {
		t.Errorf("Expected %d values, got %d", len(expected), len(results))
	}

	for i, val := range results {
		if val != expected[i] {
			t.Errorf("At index %d: expected %s, got %s", i, expected[i], val)
		}
	}
}

func TestBuffer(t *testing.T) {
	ctx := context.Background()
	input := NewStream[int](ctx, 10)

	// Buffer size 3
	output := Buffer(ctx, input, 3)

	// Send values
	go func() {
		for i := 0; i < 7; i++ {
			input.Send(i)
		}
		input.Close()
	}()

	// Collect buffers
	var batches [][]int
	for batch := range output.Chan() {
		batches = append(batches, batch)
	}

	// Should have 3 batches: [0,1,2], [3,4,5], [6]
	if len(batches) != 3 {
		t.Errorf("Expected 3 batches, got %d", len(batches))
	}

	if len(batches[0]) != 3 {
		t.Errorf("Expected first batch size 3, got %d", len(batches[0]))
	}

	if len(batches[2]) != 1 {
		t.Errorf("Expected last batch size 1, got %d", len(batches[2]))
	}
}
