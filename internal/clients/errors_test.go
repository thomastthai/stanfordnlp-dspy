package clients

import (
	"testing"
	"time"
)

func TestClientError(t *testing.T) {
	err := NewClientError(429, "Rate limit exceeded", "rate_limit_error", true)

	if err.StatusCode != 429 {
		t.Errorf("got status code %d, want 429", err.StatusCode)
	}

	if !err.IsRetryable() {
		t.Error("expected error to be retryable")
	}

	if err.Type != "rate_limit_error" {
		t.Errorf("got type %q, want %q", err.Type, "rate_limit_error")
	}

	expectedMsg := "rate_limit_error: Rate limit exceeded (status: 429)"
	if err.Error() != expectedMsg {
		t.Errorf("got error message %q, want %q", err.Error(), expectedMsg)
	}
}

func TestIsRateLimitError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{
			name:     "rate limit error by status",
			err:      NewClientError(429, "Too many requests", "", true),
			expected: true,
		},
		{
			name:     "rate limit error by type",
			err:      NewClientError(400, "Rate limited", "rate_limit_error", true),
			expected: true,
		},
		{
			name:     "not a rate limit error",
			err:      NewClientError(400, "Bad request", "invalid_request", false),
			expected: false,
		},
		{
			name:     "non-client error",
			err:      &ClientError{StatusCode: 500},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsRateLimitError(tt.err)
			if result != tt.expected {
				t.Errorf("got %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestIsServerError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{
			name:     "500 error",
			err:      NewClientError(500, "Internal server error", "", true),
			expected: true,
		},
		{
			name:     "503 error",
			err:      NewClientError(503, "Service unavailable", "", true),
			expected: true,
		},
		{
			name:     "400 error",
			err:      NewClientError(400, "Bad request", "", false),
			expected: false,
		},
		{
			name:     "429 error",
			err:      NewClientError(429, "Rate limit", "", true),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsServerError(tt.err)
			if result != tt.expected {
				t.Errorf("got %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestRetryConfig(t *testing.T) {
	config := DefaultRetryConfig()

	if config.MaxRetries != 3 {
		t.Errorf("got MaxRetries %d, want 3", config.MaxRetries)
	}

	if config.InitialWait != 1*time.Second {
		t.Errorf("got InitialWait %v, want 1s", config.InitialWait)
	}

	if config.Multiplier != 2.0 {
		t.Errorf("got Multiplier %f, want 2.0", config.Multiplier)
	}
}

func TestRetryConfigGetWaitDuration(t *testing.T) {
	config := RetryConfig{
		InitialWait: 1 * time.Second,
		MaxWait:     30 * time.Second,
		Multiplier:  2.0,
	}

	tests := []struct {
		attempt  int
		expected time.Duration
	}{
		{0, 1 * time.Second},
		{1, 2 * time.Second},
		{2, 4 * time.Second},
		{3, 8 * time.Second},
		{4, 16 * time.Second},
		{5, 30 * time.Second},  // Capped at MaxWait
		{10, 30 * time.Second}, // Still capped
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			duration := config.GetWaitDuration(tt.attempt)
			if duration != tt.expected {
				t.Errorf("attempt %d: got %v, want %v", tt.attempt, duration, tt.expected)
			}
		})
	}
}

func TestRetryConfigShouldRetryError(t *testing.T) {
	config := DefaultRetryConfig()

	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{
			name:     "rate limit error",
			err:      NewClientError(429, "Rate limited", "rate_limit_error", true),
			expected: true,
		},
		{
			name:     "server error",
			err:      NewClientError(500, "Internal error", "", true),
			expected: true,
		},
		{
			name:     "client error",
			err:      NewClientError(400, "Bad request", "", false),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := config.ShouldRetryError(tt.err)
			if result != tt.expected {
				t.Errorf("got %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestRetryConfigCustomShouldRetry(t *testing.T) {
	config := RetryConfig{
		ShouldRetry: func(err error) bool {
			// Custom logic: only retry 503 errors
			if clientErr, ok := err.(*ClientError); ok {
				return clientErr.StatusCode == 503
			}
			return false
		},
	}

	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{
			name:     "503 error",
			err:      NewClientError(503, "Service unavailable", "", true),
			expected: true,
		},
		{
			name:     "500 error",
			err:      NewClientError(500, "Internal error", "", true),
			expected: false,
		},
		{
			name:     "429 error",
			err:      NewClientError(429, "Rate limited", "", true),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := config.ShouldRetryError(tt.err)
			if result != tt.expected {
				t.Errorf("got %v, want %v", result, tt.expected)
			}
		})
	}
}
