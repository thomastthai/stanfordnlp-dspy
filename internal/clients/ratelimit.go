// Package clients provides rate limiting utilities for LM providers.
package clients

import (
	"context"
	"time"
)

// RateLimiter implements a token bucket rate limiter.
type RateLimiter struct {
	requestsPerMinute int
	tokens            chan struct{}
	ticker            *time.Ticker
	stopCh            chan struct{}
}

// NewRateLimiter creates a new rate limiter with the specified requests per minute.
func NewRateLimiter(requestsPerMinute int) *RateLimiter {
	if requestsPerMinute <= 0 {
		// No rate limiting
		return &RateLimiter{
			requestsPerMinute: 0,
			tokens:            make(chan struct{}, 1),
		}
	}

	// Create token bucket
	tokens := make(chan struct{}, requestsPerMinute)

	// Fill initial tokens
	for i := 0; i < requestsPerMinute; i++ {
		tokens <- struct{}{}
	}

	// Calculate refill interval
	// Distribute tokens evenly across the minute
	refillInterval := time.Minute / time.Duration(requestsPerMinute)

	rl := &RateLimiter{
		requestsPerMinute: requestsPerMinute,
		tokens:            tokens,
		ticker:            time.NewTicker(refillInterval),
		stopCh:            make(chan struct{}),
	}

	// Start token refiller
	go rl.refillTokens()

	return rl
}

// refillTokens periodically adds tokens to the bucket.
func (rl *RateLimiter) refillTokens() {
	for {
		select {
		case <-rl.ticker.C:
			// Try to add a token (non-blocking)
			select {
			case rl.tokens <- struct{}{}:
			default:
				// Bucket is full, skip
			}
		case <-rl.stopCh:
			return
		}
	}
}

// Wait blocks until a token is available or the context is cancelled.
func (rl *RateLimiter) Wait(ctx context.Context) error {
	// No rate limiting
	if rl.requestsPerMinute <= 0 {
		return nil
	}

	select {
	case <-rl.tokens:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Stop stops the rate limiter and releases resources.
func (rl *RateLimiter) Stop() {
	if rl.ticker != nil {
		rl.ticker.Stop()
	}
	if rl.stopCh != nil {
		select {
		case <-rl.stopCh:
			// Already closed
		default:
			close(rl.stopCh)
		}
	}
}

// SetRate updates the rate limit.
// This creates a new ticker and restarts the refill process.
func (rl *RateLimiter) SetRate(requestsPerMinute int) {
	// Stop existing ticker
	if rl.ticker != nil {
		rl.ticker.Stop()
	}

	if requestsPerMinute <= 0 {
		rl.requestsPerMinute = 0
		return
	}

	// Update rate
	rl.requestsPerMinute = requestsPerMinute

	// Create new ticker
	refillInterval := time.Minute / time.Duration(requestsPerMinute)
	rl.ticker = time.NewTicker(refillInterval)

	// Restart refiller
	go rl.refillTokens()
}

// RequestsPerMinute returns the current rate limit.
func (rl *RateLimiter) RequestsPerMinute() int {
	return rl.requestsPerMinute
}
