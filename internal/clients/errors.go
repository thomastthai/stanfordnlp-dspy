// Package clients provides error handling utilities for LM providers.
package clients

import (
	"fmt"
	"time"
)

// ClientError represents an error from a language model client.
type ClientError struct {
	// StatusCode is the HTTP status code (if applicable)
	StatusCode int

	// Message is the error message
	Message string

	// Type is the error type (e.g., "rate_limit_error", "invalid_request_error")
	Type string

	// Retryable indicates if the error can be retried
	Retryable bool

	// RetryAfter indicates how long to wait before retrying (if applicable)
	RetryAfter time.Duration
}

// Error implements the error interface.
func (e *ClientError) Error() string {
	if e.Type != "" {
		return fmt.Sprintf("%s: %s (status: %d)", e.Type, e.Message, e.StatusCode)
	}
	return fmt.Sprintf("%s (status: %d)", e.Message, e.StatusCode)
}

// IsRetryable returns true if the error can be retried.
func (e *ClientError) IsRetryable() bool {
	return e.Retryable
}

// NewClientError creates a new ClientError.
func NewClientError(statusCode int, message, errorType string, retryable bool) *ClientError {
	return &ClientError{
		StatusCode: statusCode,
		Message:    message,
		Type:       errorType,
		Retryable:  retryable,
	}
}

// IsRateLimitError checks if the error is a rate limit error.
func IsRateLimitError(err error) bool {
	if clientErr, ok := err.(*ClientError); ok {
		return clientErr.StatusCode == 429 || clientErr.Type == "rate_limit_error"
	}
	return false
}

// IsServerError checks if the error is a server error (5xx).
func IsServerError(err error) bool {
	if clientErr, ok := err.(*ClientError); ok {
		return clientErr.StatusCode >= 500 && clientErr.StatusCode < 600
	}
	return false
}

// RetryConfig configures retry behavior with exponential backoff.
type RetryConfig struct {
	// MaxRetries is the maximum number of retry attempts
	MaxRetries int

	// InitialWait is the initial wait duration before the first retry
	InitialWait time.Duration

	// MaxWait is the maximum wait duration between retries
	MaxWait time.Duration

	// Multiplier is the backoff multiplier (e.g., 2.0 for exponential backoff)
	Multiplier float64

	// ShouldRetry is a custom function to determine if an error should be retried
	ShouldRetry func(error) bool
}

// DefaultRetryConfig returns a default retry configuration.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:  3,
		InitialWait: 1 * time.Second,
		MaxWait:     30 * time.Second,
		Multiplier:  2.0,
		ShouldRetry: func(err error) bool {
			return IsRateLimitError(err) || IsServerError(err)
		},
	}
}

// GetWaitDuration calculates the wait duration for a retry attempt.
func (rc *RetryConfig) GetWaitDuration(attempt int) time.Duration {
	if attempt <= 0 {
		return rc.InitialWait
	}

	// Calculate exponential backoff
	wait := float64(rc.InitialWait)
	for i := 0; i < attempt; i++ {
		wait *= rc.Multiplier
	}

	duration := time.Duration(wait)
	if duration > rc.MaxWait {
		return rc.MaxWait
	}

	return duration
}

// ShouldRetryError checks if an error should be retried based on the configuration.
func (rc *RetryConfig) ShouldRetryError(err error) bool {
	if rc.ShouldRetry != nil {
		return rc.ShouldRetry(err)
	}

	// Default behavior: retry rate limits and server errors
	return IsRateLimitError(err) || IsServerError(err)
}
