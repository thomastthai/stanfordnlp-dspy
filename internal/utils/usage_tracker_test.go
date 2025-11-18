package utils

import (
	"testing"
	"time"
)

func TestUsageTracker(t *testing.T) {
	tracker := NewUsageTracker()

	// Record some usage
	tracker.Record("gpt-4", 100, 50, 0.01)
	tracker.Record("gpt-3.5", 200, 100, 0.005)

	// Check stats
	stats := tracker.GetStats()
	if stats.TotalTokens != 450 {
		t.Errorf("Expected 450 total tokens, got %d", stats.TotalTokens)
	}
	if stats.PromptTokens != 300 {
		t.Errorf("Expected 300 prompt tokens, got %d", stats.PromptTokens)
	}
	if stats.CompletionTokens != 150 {
		t.Errorf("Expected 150 completion tokens, got %d", stats.CompletionTokens)
	}
	if stats.TotalCost != 0.015 {
		t.Errorf("Expected 0.015 total cost, got %f", stats.TotalCost)
	}
	if stats.RequestCount != 2 {
		t.Errorf("Expected 2 requests, got %d", stats.RequestCount)
	}

	// Check history
	history := tracker.GetHistory()
	if len(history) != 2 {
		t.Errorf("Expected 2 history records, got %d", len(history))
	}

	if history[0].Model != "gpt-4" {
		t.Errorf("Expected first model to be 'gpt-4', got '%s'", history[0].Model)
	}

	// Test reset
	tracker.Reset()
	stats = tracker.GetStats()
	if stats.TotalTokens != 0 {
		t.Errorf("Expected 0 tokens after reset, got %d", stats.TotalTokens)
	}
}

func TestCalculateCost(t *testing.T) {
	tests := []struct {
		name              string
		promptTokens      int
		completionTokens  int
		promptPrice       float64
		completionPrice   float64
		expectedCost      float64
	}{
		{
			name:             "GPT-4 pricing",
			promptTokens:     1000,
			completionTokens: 500,
			promptPrice:      0.03,  // $0.03 per 1K tokens
			completionPrice:  0.06,  // $0.06 per 1K tokens
			expectedCost:     0.06,  // (1000/1000)*0.03 + (500/1000)*0.06 = 0.03 + 0.03 = 0.06
		},
		{
			name:             "GPT-3.5 pricing",
			promptTokens:     2000,
			completionTokens: 1000,
			promptPrice:      0.001,
			completionPrice:  0.002,
			expectedCost:     0.004, // (2000/1000)*0.001 + (1000/1000)*0.002 = 0.002 + 0.002 = 0.004
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cost := CalculateCost(tt.promptTokens, tt.completionTokens, tt.promptPrice, tt.completionPrice)
			if cost != tt.expectedCost {
				t.Errorf("CalculateCost() = %f, want %f", cost, tt.expectedCost)
			}
		})
	}
}

func TestUsageTrackerConcurrency(t *testing.T) {
	tracker := NewUsageTracker()

	// Record concurrently
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			tracker.Record("test-model", 100, 50, 0.01)
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	stats := tracker.GetStats()
	if stats.RequestCount != 10 {
		t.Errorf("Expected 10 requests, got %d", stats.RequestCount)
	}
	if stats.TotalTokens != 1500 {
		t.Errorf("Expected 1500 total tokens, got %d", stats.TotalTokens)
	}
}

func TestUsageStats_Duration(t *testing.T) {
	tracker := NewUsageTracker()
	
	// Wait a bit
	time.Sleep(100 * time.Millisecond)
	
	stats := tracker.GetStats()
	duration := stats.EndTime.Sub(stats.StartTime)
	if duration < 100*time.Millisecond {
		t.Errorf("Expected duration >= 100ms, got %v", duration)
	}
}
