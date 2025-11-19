package bedrock

import (
	"testing"
)

func TestProvider_Name(t *testing.T) {
	p := &Provider{}
	if p.Name() != "bedrock" {
		t.Errorf("expected provider name 'bedrock', got %s", p.Name())
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
		"anthropic.claude-3-opus-20240229-v1:0",
		"amazon.titan-text-express-v1",
		"meta.llama2-70b-chat-v1",
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

func TestGetModelProvider(t *testing.T) {
	tests := []struct {
		modelID  string
		expected string
	}{
		{"anthropic.claude-3-opus-20240229-v1:0", "anthropic"},
		{"amazon.titan-text-express-v1", "titan"},
		{"meta.llama2-70b-chat-v1", "llama"},
		{"ai21.j2-ultra-v1", "ai21"},
		{"cohere.command-text-v14", "cohere"},
		{"unknown-model", "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			result := getModelProvider(tt.modelID)
			if result != tt.expected {
				t.Errorf("getModelProvider(%s) = %s, expected %s", tt.modelID, result, tt.expected)
			}
		})
	}
}
