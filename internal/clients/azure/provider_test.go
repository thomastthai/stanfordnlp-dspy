package azure

import (
	"testing"
)

func TestProvider_Name(t *testing.T) {
	p := &Provider{}
	if p.Name() != "azure" {
		t.Errorf("expected provider name 'azure', got %s", p.Name())
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
		"gpt-4o",
		"gpt-4",
		"gpt-35-turbo",
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
