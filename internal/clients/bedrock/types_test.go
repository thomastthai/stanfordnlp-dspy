package bedrock

import (
	"testing"
)

func TestGetModelCapabilities(t *testing.T) {
	tests := []struct {
		name       string
		family     ModelFamily
		wantChat   bool
		wantStream bool
	}{
		{
			name:       "anthropic",
			family:     ModelFamilyAnthropic,
			wantChat:   true,
			wantStream: true,
		},
		{
			name:       "titan",
			family:     ModelFamilyTitan,
			wantChat:   false,
			wantStream: true,
		},
		{
			name:       "llama",
			family:     ModelFamilyLlama,
			wantChat:   true,
			wantStream: true,
		},
		{
			name:       "ai21",
			family:     ModelFamilyAI21,
			wantChat:   false,
			wantStream: false,
		},
		{
			name:       "cohere",
			family:     ModelFamilyCohere,
			wantChat:   false,
			wantStream: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			caps := GetModelCapabilities(tt.family)

			if caps.ChatCompletion != tt.wantChat {
				t.Errorf("ChatCompletion = %v, want %v", caps.ChatCompletion, tt.wantChat)
			}

			if caps.Streaming != tt.wantStream {
				t.Errorf("Streaming = %v, want %v", caps.Streaming, tt.wantStream)
			}
		})
	}
}

func TestDefaultThrottlingConfig(t *testing.T) {
	config := DefaultThrottlingConfig()

	if config.MaxRetries <= 0 {
		t.Error("expected positive MaxRetries")
	}

	if config.InitialBackoff <= 0 {
		t.Error("expected positive InitialBackoff")
	}

	if config.MaxBackoff <= config.InitialBackoff {
		t.Error("expected MaxBackoff > InitialBackoff")
	}

	if config.BackoffMultiplier <= 1.0 {
		t.Error("expected BackoffMultiplier > 1.0")
	}
}

func TestBedrockError_Error(t *testing.T) {
	err := &BedrockError{
		StatusCode: 429,
		Code:       "ThrottlingException",
		Message:    "Rate limit exceeded",
		Retryable:  true,
	}

	msg := err.Error()
	if msg != "Rate limit exceeded" {
		t.Errorf("expected 'Rate limit exceeded', got '%s'", msg)
	}
}

func TestValidationError_Error(t *testing.T) {
	err := &ValidationError{
		Field:   "temperature",
		Message: "temperature must be between 0 and 1",
	}

	msg := err.Error()
	if msg != "temperature must be between 0 and 1" {
		t.Errorf("expected 'temperature must be between 0 and 1', got '%s'", msg)
	}
}
