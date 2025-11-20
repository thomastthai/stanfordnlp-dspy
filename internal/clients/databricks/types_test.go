package databricks

import (
	"testing"
)

func TestDatabricksError_Error(t *testing.T) {
	tests := []struct {
		name string
		err  *DatabricksError
		want string
	}{
		{
			name: "error with details",
			err: &DatabricksError{
				ErrorCode: "RESOURCE_EXHAUSTED",
				Message:   "Rate limit exceeded",
				Details:   "Try again later",
			},
			want: "Rate limit exceeded: Try again later",
		},
		{
			name: "error without details",
			err: &DatabricksError{
				ErrorCode: "INVALID_REQUEST",
				Message:   "Invalid request",
			},
			want: "Invalid request",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.err.Error(); got != tt.want {
				t.Errorf("Error() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDatabricksError_IsRetryable(t *testing.T) {
	tests := []struct {
		name string
		code string
		want bool
	}{
		{
			name: "resource exhausted",
			code: "RESOURCE_EXHAUSTED",
			want: true,
		},
		{
			name: "temporarily unavailable",
			code: "TEMPORARILY_UNAVAILABLE",
			want: true,
		},
		{
			name: "internal error",
			code: "INTERNAL_ERROR",
			want: true,
		},
		{
			name: "invalid request",
			code: "INVALID_REQUEST",
			want: false,
		},
		{
			name: "unauthenticated",
			code: "UNAUTHENTICATED",
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := &DatabricksError{ErrorCode: tt.code}
			if got := err.IsRetryable(); got != tt.want {
				t.Errorf("IsRetryable() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetModelCapabilities(t *testing.T) {
	tests := []struct {
		name       string
		model      FoundationModel
		wantChat   bool
		wantStream bool
	}{
		{
			name:       "DBRX Instruct",
			model:      ModelDBRXInstruct,
			wantChat:   true,
			wantStream: true,
		},
		{
			name:       "Llama 3 70B",
			model:      ModelLlama3_70BInstruct,
			wantChat:   true,
			wantStream: true,
		},
		{
			name:       "Mixtral",
			model:      ModelMixtral8x7BInstruct,
			wantChat:   true,
			wantStream: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			caps := GetModelCapabilities(tt.model)

			if caps.ChatCompletion != tt.wantChat {
				t.Errorf("ChatCompletion = %v, want %v", caps.ChatCompletion, tt.wantChat)
			}

			if caps.Streaming != tt.wantStream {
				t.Errorf("Streaming = %v, want %v", caps.Streaming, tt.wantStream)
			}
		})
	}
}
