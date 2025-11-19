package anthropic

import (
	"testing"
)

func TestProvider_Name(t *testing.T) {
	p := &Provider{}
	if p.Name() != "anthropic" {
		t.Errorf("expected provider name 'anthropic', got %s", p.Name())
	}
}

func TestProvider_SupportedModels(t *testing.T) {
	p := &Provider{}
	models := p.SupportedModels()

	if len(models) == 0 {
		t.Error("expected supported models, got empty list")
	}

	// Check for key models
	expectedModels := []string{
		"claude-3-opus-20240229",
		"claude-3-5-sonnet-20240620",
		"claude-3-haiku-20240307",
	}

	for _, expected := range expectedModels {
		found := false
		for _, model := range models {
			if model == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected model %s not found in supported models", expected)
		}
	}
}

func TestIsModelSupported(t *testing.T) {
	tests := []struct {
		model    string
		expected bool
	}{
		{"claude-3-opus-20240229", true},
		{"claude-3-sonnet", true},
		{"claude-3-haiku", true},
		{"claude-2.1", true},
		{"gpt-4", false},
		{"invalid-model", false},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			result := isModelSupported(tt.model)
			if result != tt.expected {
				t.Errorf("isModelSupported(%s) = %v, expected %v", tt.model, result, tt.expected)
			}
		})
	}
}

func TestGetModelInfo(t *testing.T) {
	tests := []struct {
		model               string
		expectedContext     int
		expectedSupportsVis bool
	}{
		{"claude-3-opus-20240229", 200000, true},
		{"claude-3-5-sonnet-20240620", 200000, true},
		{"claude-3-haiku-20240307", 200000, true},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			info := GetModelInfo(tt.model)
			if info.ContextWindow != tt.expectedContext {
				t.Errorf("expected context window %d, got %d", tt.expectedContext, info.ContextWindow)
			}
			if info.SupportsVision != tt.expectedSupportsVis {
				t.Errorf("expected SupportsVision %v, got %v", tt.expectedSupportsVis, info.SupportsVision)
			}
		})
	}
}

func TestCountTokens(t *testing.T) {
	tests := []struct {
		text     string
		expected int
	}{
		{"", 0},
		{"hello", 1},
		{"hello world", 2},
		{"this is a longer test string", 7},
	}

	for _, tt := range tests {
		t.Run(tt.text, func(t *testing.T) {
			result := CountTokens(tt.text)
			// Allow some flexibility in token counting
			if result < tt.expected-1 || result > tt.expected+1 {
				t.Errorf("CountTokens(%q) = %d, expected ~%d", tt.text, result, tt.expected)
			}
		})
	}
}
