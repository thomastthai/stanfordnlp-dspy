package utils

import (
	"sync"
	"time"
)

// UsageStats represents usage statistics.
type UsageStats struct {
	TotalTokens      int
	PromptTokens     int
	CompletionTokens int
	TotalCost        float64
	RequestCount     int
	StartTime        time.Time
	EndTime          time.Time
}

// UsageTracker tracks token usage and costs for LM calls.
type UsageTracker struct {
	mu               sync.RWMutex
	totalTokens      int
	promptTokens     int
	completionTokens int
	totalCost        float64
	requestCount     int
	startTime        time.Time
	history          []UsageRecord
}

// UsageRecord represents a single usage record.
type UsageRecord struct {
	Timestamp        time.Time
	Model            string
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	Cost             float64
}

// NewUsageTracker creates a new usage tracker.
func NewUsageTracker() *UsageTracker {
	return &UsageTracker{
		startTime: time.Now(),
		history:   make([]UsageRecord, 0),
	}
}

// Record records usage for a single request.
func (ut *UsageTracker) Record(model string, promptTokens, completionTokens int, cost float64) {
	ut.mu.Lock()
	defer ut.mu.Unlock()

	totalTokens := promptTokens + completionTokens

	ut.promptTokens += promptTokens
	ut.completionTokens += completionTokens
	ut.totalTokens += totalTokens
	ut.totalCost += cost
	ut.requestCount++

	record := UsageRecord{
		Timestamp:        time.Now(),
		Model:            model,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      totalTokens,
		Cost:             cost,
	}

	ut.history = append(ut.history, record)
}

// GetStats returns current usage statistics.
func (ut *UsageTracker) GetStats() UsageStats {
	ut.mu.RLock()
	defer ut.mu.RUnlock()

	return UsageStats{
		TotalTokens:      ut.totalTokens,
		PromptTokens:     ut.promptTokens,
		CompletionTokens: ut.completionTokens,
		TotalCost:        ut.totalCost,
		RequestCount:     ut.requestCount,
		StartTime:        ut.startTime,
		EndTime:          time.Now(),
	}
}

// GetHistory returns the usage history.
func (ut *UsageTracker) GetHistory() []UsageRecord {
	ut.mu.RLock()
	defer ut.mu.RUnlock()

	// Return a copy to prevent external modification
	history := make([]UsageRecord, len(ut.history))
	copy(history, ut.history)
	return history
}

// Reset resets the usage tracker.
func (ut *UsageTracker) Reset() {
	ut.mu.Lock()
	defer ut.mu.Unlock()

	ut.totalTokens = 0
	ut.promptTokens = 0
	ut.completionTokens = 0
	ut.totalCost = 0
	ut.requestCount = 0
	ut.startTime = time.Now()
	ut.history = make([]UsageRecord, 0)
}

// CalculateCost calculates the cost based on token usage and pricing.
// Pricing is per 1K tokens.
func CalculateCost(promptTokens, completionTokens int, promptPrice, completionPrice float64) float64 {
	promptCost := float64(promptTokens) / 1000.0 * promptPrice
	completionCost := float64(completionTokens) / 1000.0 * completionPrice
	return promptCost + completionCost
}
