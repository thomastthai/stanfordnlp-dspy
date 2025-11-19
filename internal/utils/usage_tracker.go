package utils

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// UsageEntry represents a single usage entry for an LM call.
type UsageEntry struct {
	PromptTokens     int                    `json:"prompt_tokens"`
	CompletionTokens int                    `json:"completion_tokens"`
	TotalTokens      int                    `json:"total_tokens"`
	EstimatedCost    float64                `json:"estimated_cost,omitempty"`
	Model            string                 `json:"model,omitempty"`
	Metadata         map[string]interface{} `json:"metadata,omitempty"`
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
	usageByModel map[string][]*UsageEntry
	mu           sync.RWMutex
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
		usageByModel: make(map[string][]*UsageEntry),
	}
}

// AddUsage adds a usage entry for the specified model.
func (ut *UsageTracker) AddUsage(model string, entry *UsageEntry) {
	ut.mu.Lock()
	defer ut.mu.Unlock()
	
	if entry.Model == "" {
		entry.Model = model
	}
	
	ut.usageByModel[model] = append(ut.usageByModel[model], entry)
}

// GetUsage returns all usage entries for the specified model.
func (ut *UsageTracker) GetUsage(model string) []*UsageEntry {
	ut.mu.RLock()
	defer ut.mu.RUnlock()
	
	entries, ok := ut.usageByModel[model]
	if !ok {
		return []*UsageEntry{}
	}
	
	// Return a copy
	result := make([]*UsageEntry, len(entries))
	copy(result, entries)
	return result
}

// GetTotalUsage returns the total usage across all models.
func (ut *UsageTracker) GetTotalUsage() map[string]*UsageEntry {
	ut.mu.RLock()
	defer ut.mu.RUnlock()
	
	totals := make(map[string]*UsageEntry)
	
	for model, entries := range ut.usageByModel {
		total := &UsageEntry{
			Model: model,
		}
		
		for _, entry := range entries {
			total.PromptTokens += entry.PromptTokens
			total.CompletionTokens += entry.CompletionTokens
			total.TotalTokens += entry.TotalTokens
			total.EstimatedCost += entry.EstimatedCost
		}
		
		totals[model] = total
	}
	
	return totals
}

// GetGrandTotal returns the grand total usage across all models.
func (ut *UsageTracker) GetGrandTotal() *UsageEntry {
	totals := ut.GetTotalUsage()
	
	grand := &UsageEntry{}
	for _, total := range totals {
		grand.PromptTokens += total.PromptTokens
		grand.CompletionTokens += total.CompletionTokens
		grand.TotalTokens += total.TotalTokens
		grand.EstimatedCost += total.EstimatedCost
	}
	
	return grand
}

// Reset resets all usage tracking.
func (ut *UsageTracker) Reset() {
	ut.mu.Lock()
	defer ut.mu.Unlock()
	
	ut.usageByModel = make(map[string][]*UsageEntry)
}

// Export exports usage statistics to JSON.
func (ut *UsageTracker) Export() ([]byte, error) {
	ut.mu.RLock()
	defer ut.mu.RUnlock()
	
	data := map[string]interface{}{
		"usage_by_model": ut.usageByModel,
		"totals":         ut.GetTotalUsage(),
		"grand_total":    ut.GetGrandTotal(),
	}
	
	return json.MarshalIndent(data, "", "  ")
}

// ExportToFile exports usage statistics to a JSON file.
func (ut *UsageTracker) ExportToFile(path string) error {
	data, err := ut.Export()
	if err != nil {
		return fmt.Errorf("failed to export usage: %w", err)
	}
	
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}
	
	return nil
}

// Import imports usage statistics from JSON.
func (ut *UsageTracker) Import(data []byte) error {
	ut.mu.Lock()
	defer ut.mu.Unlock()
	
	var imported map[string]interface{}
	if err := json.Unmarshal(data, &imported); err != nil {
		return fmt.Errorf("failed to unmarshal usage data: %w", err)
	}
	
	usageByModel, ok := imported["usage_by_model"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid usage data format")
	}
	
	ut.usageByModel = make(map[string][]*UsageEntry)
	for model, entries := range usageByModel {
		entryList, ok := entries.([]interface{})
		if !ok {
			continue
		}
		
		ut.usageByModel[model] = make([]*UsageEntry, 0, len(entryList))
		for _, e := range entryList {
			entryMap, ok := e.(map[string]interface{})
			if !ok {
				continue
			}
			
			entry := &UsageEntry{}
			if v, ok := entryMap["prompt_tokens"].(float64); ok {
				entry.PromptTokens = int(v)
			}
			if v, ok := entryMap["completion_tokens"].(float64); ok {
				entry.CompletionTokens = int(v)
			}
			if v, ok := entryMap["total_tokens"].(float64); ok {
				entry.TotalTokens = int(v)
			}
			if v, ok := entryMap["estimated_cost"].(float64); ok {
				entry.EstimatedCost = v
			}
			if v, ok := entryMap["model"].(string); ok {
				entry.Model = v
			}
			
			ut.usageByModel[model] = append(ut.usageByModel[model], entry)
		}
	}
	
	return nil
}

// ModelPricing contains pricing information for a model.
type ModelPricing struct {
	PromptCostPer1K     float64
	CompletionCostPer1K float64
}

// DefaultPricing returns default pricing for common models.
// Prices are approximate and should be updated based on actual pricing.
func DefaultPricing() map[string]ModelPricing {
	return map[string]ModelPricing{
		"gpt-4": {
			PromptCostPer1K:     0.03,
			CompletionCostPer1K: 0.06,
		},
		"gpt-4-turbo": {
			PromptCostPer1K:     0.01,
			CompletionCostPer1K: 0.03,
		},
		"gpt-3.5-turbo": {
			PromptCostPer1K:     0.0005,
			CompletionCostPer1K: 0.0015,
		},
		"claude-3-opus": {
			PromptCostPer1K:     0.015,
			CompletionCostPer1K: 0.075,
		},
		"claude-3-sonnet": {
			PromptCostPer1K:     0.003,
			CompletionCostPer1K: 0.015,
		},
	}
}

// EstimateCost estimates the cost for a usage entry based on pricing.
func EstimateCost(entry *UsageEntry, pricing ModelPricing) float64 {
	promptCost := float64(entry.PromptTokens) / 1000.0 * pricing.PromptCostPer1K
	completionCost := float64(entry.CompletionTokens) / 1000.0 * pricing.CompletionCostPer1K
	return promptCost + completionCost
}

// UpdateCosts updates estimated costs for all entries based on pricing.
func (ut *UsageTracker) UpdateCosts(pricing map[string]ModelPricing) {
	ut.mu.Lock()
	defer ut.mu.Unlock()
	
	for model, entries := range ut.usageByModel {
		if modelPricing, ok := pricing[model]; ok {
			for _, entry := range entries {
				entry.EstimatedCost = EstimateCost(entry, modelPricing)
			}
		}
	}
}
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
