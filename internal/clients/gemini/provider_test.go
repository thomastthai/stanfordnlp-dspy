package gemini

import (
	"testing"
)

func TestProvider_Name(t *testing.T) {
	p := &Provider{}
	if p.Name() != "gemini" {
		t.Errorf("expected provider name 'gemini', got %s", p.Name())
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
		"gemini-1.5-pro",
		"gemini-1.5-flash",
		"gemini-pro",
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
		{"gemini-1.5-pro", true},
		{"gemini-1.5-flash", true},
		{"gemini-pro", true},
		{"gemini-pro-vision", true},
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
