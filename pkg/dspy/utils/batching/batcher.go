// Package batching provides request batching utilities for DSPy.
package batching

import (
	"context"
	"sync"
	"time"
)

// Request represents a single request to be batched.
type Request struct {
	ID      string
	Data    interface{}
	Result  chan Result
	Context context.Context
}

// Result represents the result of a batched request.
type Result struct {
	Data  interface{}
	Error error
}

// ProcessFunc processes a batch of requests.
type ProcessFunc func(ctx context.Context, requests []Request) ([]Result, error)

// Batcher batches requests for efficient processing.
type Batcher struct {
	mu             sync.Mutex
	requests       []Request
	maxBatchSize   int
	maxWaitTime    time.Duration
	processFn      ProcessFunc
	timer          *time.Timer
	stopCh         chan struct{}
	wg             sync.WaitGroup
}

// Config configures the batcher.
type Config struct {
	MaxBatchSize int
	MaxWaitTime  time.Duration
	ProcessFunc  ProcessFunc
}

// NewBatcher creates a new request batcher.
func NewBatcher(config Config) *Batcher {
	if config.MaxBatchSize <= 0 {
		config.MaxBatchSize = 10
	}
	if config.MaxWaitTime <= 0 {
		config.MaxWaitTime = 100 * time.Millisecond
	}
	
	b := &Batcher{
		requests:     make([]Request, 0, config.MaxBatchSize),
		maxBatchSize: config.MaxBatchSize,
		maxWaitTime:  config.MaxWaitTime,
		processFn:    config.ProcessFunc,
		stopCh:       make(chan struct{}),
	}
	
	return b
}

// Submit submits a request for batching.
func (b *Batcher) Submit(ctx context.Context, data interface{}) (interface{}, error) {
	req := Request{
		Data:    data,
		Result:  make(chan Result, 1),
		Context: ctx,
	}
	
	b.mu.Lock()
	
	// Add request to batch
	b.requests = append(b.requests, req)
	
	// Check if batch is full
	if len(b.requests) >= b.maxBatchSize {
		// Process immediately
		batch := b.requests
		b.requests = make([]Request, 0, b.maxBatchSize)
		
		// Stop timer if running
		if b.timer != nil {
			b.timer.Stop()
			b.timer = nil
		}
		
		b.mu.Unlock()
		
		// Process batch asynchronously
		go b.processBatch(batch)
	} else {
		// Start timer if not already running
		if b.timer == nil {
			b.timer = time.AfterFunc(b.maxWaitTime, func() {
				b.mu.Lock()
				if len(b.requests) > 0 {
					batch := b.requests
					b.requests = make([]Request, 0, b.maxBatchSize)
					b.timer = nil
					b.mu.Unlock()
					
					go b.processBatch(batch)
				} else {
					b.mu.Unlock()
				}
			})
		}
		
		b.mu.Unlock()
	}
	
	// Wait for result
	select {
	case result := <-req.Result:
		return result.Data, result.Error
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// processBatch processes a batch of requests.
func (b *Batcher) processBatch(batch []Request) {
	b.wg.Add(1)
	defer b.wg.Done()
	
	if len(batch) == 0 {
		return
	}
	
	// Use first request's context as batch context
	ctx := batch[0].Context
	
	// Process batch
	results, err := b.processFn(ctx, batch)
	
	// Handle global error
	if err != nil {
		for _, req := range batch {
			req.Result <- Result{Error: err}
		}
		return
	}
	
	// Send individual results
	for i, req := range batch {
		if i < len(results) {
			req.Result <- results[i]
		} else {
			req.Result <- Result{Error: ErrNoResult}
		}
	}
}

// Close closes the batcher and waits for pending batches.
func (b *Batcher) Close() {
	close(b.stopCh)
	
	// Process remaining requests
	b.mu.Lock()
	if len(b.requests) > 0 {
		batch := b.requests
		b.requests = nil
		b.mu.Unlock()
		
		b.processBatch(batch)
	} else {
		b.mu.Unlock()
	}
	
	// Wait for all batches to complete
	b.wg.Wait()
}

// ErrNoResult is returned when no result is available for a request.
var ErrNoResult = &NoResultError{}

// NoResultError indicates no result was returned for a request.
type NoResultError struct{}

func (e *NoResultError) Error() string {
	return "no result available"
}

// AutoBatcher automatically batches requests based on configuration.
type AutoBatcher struct {
	batcher       *Batcher
	fallbackFn    func(context.Context, interface{}) (interface{}, error)
	enabled       bool
}

// NewAutoBatcher creates an auto-batcher that falls back to individual processing.
func NewAutoBatcher(config Config, fallbackFn func(context.Context, interface{}) (interface{}, error)) *AutoBatcher {
	return &AutoBatcher{
		batcher:    NewBatcher(config),
		fallbackFn: fallbackFn,
		enabled:    true,
	}
}

// Process processes a request, using batching if enabled.
func (a *AutoBatcher) Process(ctx context.Context, data interface{}) (interface{}, error) {
	if a.enabled {
		return a.batcher.Submit(ctx, data)
	}
	return a.fallbackFn(ctx, data)
}

// Enable enables batching.
func (a *AutoBatcher) Enable() {
	a.enabled = true
}

// Disable disables batching and uses fallback function.
func (a *AutoBatcher) Disable() {
	a.enabled = false
}

// Close closes the auto-batcher.
func (a *AutoBatcher) Close() {
	a.batcher.Close()
}
