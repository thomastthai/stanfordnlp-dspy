// Package cost provides cost tracking utilities for DSPy.
package cost

import (
	"sync"
	"time"
)

// CostTracker tracks costs for LLM operations.
type CostTracker struct {
	mu     sync.RWMutex
	costs  map[string]float64 // model -> total cost
	tokens map[string]int     // model -> total tokens
	calls  map[string]int     // model -> call count
}

// NewCostTracker creates a new cost tracker.
func NewCostTracker() *CostTracker {
	return &CostTracker{
		costs:  make(map[string]float64),
		tokens: make(map[string]int),
		calls:  make(map[string]int),
	}
}

// RecordUsage records token usage and calculates cost.
func (t *CostTracker) RecordUsage(model string, promptTokens, completionTokens int) float64 {
	cost := CalculateCost(model, promptTokens, completionTokens)
	
	t.mu.Lock()
	defer t.mu.Unlock()
	
	t.costs[model] += cost
	t.tokens[model] += promptTokens + completionTokens
	t.calls[model]++
	
	return cost
}

// GetTotalCost returns the total cost across all models.
func (t *CostTracker) GetTotalCost() float64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	
	total := 0.0
	for _, cost := range t.costs {
		total += cost
	}
	return total
}

// GetModelCost returns the cost for a specific model.
func (t *CostTracker) GetModelCost(model string) float64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	
	return t.costs[model]
}

// GetModelTokens returns the token count for a specific model.
func (t *CostTracker) GetModelTokens(model string) int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	
	return t.tokens[model]
}

// GetModelCalls returns the call count for a specific model.
func (t *CostTracker) GetModelCalls(model string) int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	
	return t.calls[model]
}

// GetReport returns a cost report.
func (t *CostTracker) GetReport() CostReport {
	t.mu.RLock()
	defer t.mu.RUnlock()
	
	models := make([]ModelCost, 0, len(t.costs))
	totalCost := 0.0
	totalTokens := 0
	totalCalls := 0
	
	for model := range t.costs {
		cost := t.costs[model]
		tokens := t.tokens[model]
		calls := t.calls[model]
		
		models = append(models, ModelCost{
			Model:  model,
			Cost:   cost,
			Tokens: tokens,
			Calls:  calls,
		})
		
		totalCost += cost
		totalTokens += tokens
		totalCalls += calls
	}
	
	return CostReport{
		Models:      models,
		TotalCost:   totalCost,
		TotalTokens: totalTokens,
		TotalCalls:  totalCalls,
		Timestamp:   time.Now(),
	}
}

// Reset clears all tracked costs.
func (t *CostTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	
	t.costs = make(map[string]float64)
	t.tokens = make(map[string]int)
	t.calls = make(map[string]int)
}

// CostReport contains cost tracking information.
type CostReport struct {
	Models      []ModelCost
	TotalCost   float64
	TotalTokens int
	TotalCalls  int
	Timestamp   time.Time
}

// ModelCost contains cost information for a specific model.
type ModelCost struct {
	Model  string
	Cost   float64
	Tokens int
	Calls  int
}

// BudgetLimiter enforces cost budgets.
type BudgetLimiter struct {
	tracker    *CostTracker
	maxCost    float64
	alertThreshold float64
	onAlert    func(current, max float64)
	onExceeded func(current, max float64)
}

// NewBudgetLimiter creates a new budget limiter.
func NewBudgetLimiter(maxCost float64, alertThreshold float64) *BudgetLimiter {
	return &BudgetLimiter{
		tracker:        NewCostTracker(),
		maxCost:        maxCost,
		alertThreshold: alertThreshold,
	}
}

// SetAlertCallback sets the callback for budget alerts.
func (b *BudgetLimiter) SetAlertCallback(fn func(current, max float64)) {
	b.onAlert = fn
}

// SetExceededCallback sets the callback for budget exceeded.
func (b *BudgetLimiter) SetExceededCallback(fn func(current, max float64)) {
	b.onExceeded = fn
}

// RecordUsage records usage and checks budget limits.
func (b *BudgetLimiter) RecordUsage(model string, promptTokens, completionTokens int) (float64, error) {
	cost := b.tracker.RecordUsage(model, promptTokens, completionTokens)
	totalCost := b.tracker.GetTotalCost()
	
	// Check if budget exceeded
	if totalCost > b.maxCost {
		if b.onExceeded != nil {
			b.onExceeded(totalCost, b.maxCost)
		}
		return cost, ErrBudgetExceeded
	}
	
	// Check alert threshold
	if totalCost > b.maxCost*b.alertThreshold && b.onAlert != nil {
		b.onAlert(totalCost, b.maxCost)
	}
	
	return cost, nil
}

// GetTracker returns the underlying cost tracker.
func (b *BudgetLimiter) GetTracker() *CostTracker {
	return b.tracker
}

// ErrBudgetExceeded is returned when the budget is exceeded.
var ErrBudgetExceeded = &BudgetExceededError{}

// BudgetExceededError indicates that the budget has been exceeded.
type BudgetExceededError struct {
	Current float64
	Max     float64
}

func (e *BudgetExceededError) Error() string {
	return "budget exceeded"
}
