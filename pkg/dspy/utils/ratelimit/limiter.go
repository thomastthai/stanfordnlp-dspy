// Package ratelimit provides rate limiting utilities for DSPy.
package ratelimit

import (
	"context"
	"time"

	"golang.org/x/time/rate"
)

// RateLimiter provides rate limiting functionality.
type RateLimiter interface {
	// Allow checks if an operation is allowed
	Allow() bool

	// Wait blocks until an operation is allowed
	Wait(ctx context.Context) error

	// Limit returns the current rate limit
	Limit() rate.Limit
}

// TokenBucketLimiter implements rate limiting using the token bucket algorithm.
type TokenBucketLimiter struct {
	limiter *rate.Limiter
}

// NewTokenBucketLimiter creates a new token bucket rate limiter.
// ratePerSecond is the number of requests per second.
// burst is the maximum number of requests that can be made at once.
func NewTokenBucketLimiter(ratePerSecond float64, burst int) *TokenBucketLimiter {
	return &TokenBucketLimiter{
		limiter: rate.NewLimiter(rate.Limit(ratePerSecond), burst),
	}
}

// Allow implements RateLimiter.Allow.
func (l *TokenBucketLimiter) Allow() bool {
	return l.limiter.Allow()
}

// Wait implements RateLimiter.Wait.
func (l *TokenBucketLimiter) Wait(ctx context.Context) error {
	return l.limiter.Wait(ctx)
}

// Limit implements RateLimiter.Limit.
func (l *TokenBucketLimiter) Limit() rate.Limit {
	return l.limiter.Limit()
}

// SetLimit updates the rate limit.
func (l *TokenBucketLimiter) SetLimit(ratePerSecond float64) {
	l.limiter.SetLimit(rate.Limit(ratePerSecond))
}

// SetBurst updates the burst size.
func (l *TokenBucketLimiter) SetBurst(burst int) {
	l.limiter.SetBurst(burst)
}

// PerModelLimiter provides per-model rate limiting.
type PerModelLimiter struct {
	limiters map[string]*TokenBucketLimiter
	defaults struct {
		rate  float64
		burst int
	}
}

// NewPerModelLimiter creates a new per-model rate limiter.
func NewPerModelLimiter(defaultRate float64, defaultBurst int) *PerModelLimiter {
	return &PerModelLimiter{
		limiters: make(map[string]*TokenBucketLimiter),
		defaults: struct {
			rate  float64
			burst int
		}{
			rate:  defaultRate,
			burst: defaultBurst,
		},
	}
}

// SetModelLimit sets a custom rate limit for a specific model.
func (l *PerModelLimiter) SetModelLimit(model string, ratePerSecond float64, burst int) {
	l.limiters[model] = NewTokenBucketLimiter(ratePerSecond, burst)
}

// GetLimiter returns the rate limiter for a specific model.
func (l *PerModelLimiter) GetLimiter(model string) RateLimiter {
	if limiter, ok := l.limiters[model]; ok {
		return limiter
	}

	// Create default limiter for model
	limiter := NewTokenBucketLimiter(l.defaults.rate, l.defaults.burst)
	l.limiters[model] = limiter
	return limiter
}

// Allow checks if a request for the given model is allowed.
func (l *PerModelLimiter) Allow(model string) bool {
	return l.GetLimiter(model).Allow()
}

// Wait blocks until a request for the given model is allowed.
func (l *PerModelLimiter) Wait(ctx context.Context, model string) error {
	return l.GetLimiter(model).Wait(ctx)
}

// RetryConfig configures retry behavior with rate limiting.
type RetryConfig struct {
	MaxRetries        int
	InitialBackoff    time.Duration
	MaxBackoff        time.Duration
	BackoffMultiplier float64
}

// DefaultRetryConfig returns default retry configuration.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:        3,
		InitialBackoff:    time.Second,
		MaxBackoff:        30 * time.Second,
		BackoffMultiplier: 2.0,
	}
}

// RetryWithBackoff retries an operation with exponential backoff.
func RetryWithBackoff(ctx context.Context, config RetryConfig, fn func() error) error {
	var err error
	backoff := config.InitialBackoff

	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		err = fn()
		if err == nil {
			return nil
		}

		// Check if we should retry
		if attempt < config.MaxRetries {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
				// Increase backoff for next attempt
				backoff = time.Duration(float64(backoff) * config.BackoffMultiplier)
				if backoff > config.MaxBackoff {
					backoff = config.MaxBackoff
				}
			}
		}
	}

	return err
}
