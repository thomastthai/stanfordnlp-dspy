package openai

import (
	"testing"
)

func TestEmbeddingModels(t *testing.T) {
	expectedModels := []string{
		"text-embedding-3-small",
		"text-embedding-3-large",
		"text-embedding-ada-002",
	}

	for _, model := range expectedModels {
		t.Run(model, func(t *testing.T) {
			info, ok := GetEmbeddingModelInfo(model)
			if !ok {
				t.Fatalf("model %q not found", model)
			}

			if info.Name != model {
				t.Errorf("got name %q, want %q", info.Name, model)
			}

			if info.Dimensions <= 0 {
				t.Errorf("dimensions should be positive, got %d", info.Dimensions)
			}

			if info.MaxTokens <= 0 {
				t.Errorf("max tokens should be positive, got %d", info.MaxTokens)
			}

			if info.CostPer1MTokens < 0 {
				t.Errorf("cost should be non-negative, got %f", info.CostPer1MTokens)
			}
		})
	}
}

func TestGetEmbeddingModelInfo(t *testing.T) {
	tests := []struct {
		name       string
		model      string
		wantFound  bool
		wantDims   int
	}{
		{
			name:      "text-embedding-3-small",
			model:     "text-embedding-3-small",
			wantFound: true,
			wantDims:  1536,
		},
		{
			name:      "text-embedding-3-large",
			model:     "text-embedding-3-large",
			wantFound: true,
			wantDims:  3072,
		},
		{
			name:      "text-embedding-ada-002",
			model:     "text-embedding-ada-002",
			wantFound: true,
			wantDims:  1536,
		},
		{
			name:      "invalid model",
			model:     "invalid-model",
			wantFound: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info, found := GetEmbeddingModelInfo(tt.model)

			if found != tt.wantFound {
				t.Errorf("got found %v, want %v", found, tt.wantFound)
			}

			if tt.wantFound && info.Dimensions != tt.wantDims {
				t.Errorf("got dimensions %d, want %d", info.Dimensions, tt.wantDims)
			}
		})
	}
}

func TestEmbeddingModelCosts(t *testing.T) {
	// Verify that costs are reasonable
	for model, info := range EmbeddingModels {
		t.Run(model, func(t *testing.T) {
			// Cost should be between $0.001 and $1.00 per 1M tokens
			if info.CostPer1MTokens < 0.001 || info.CostPer1MTokens > 1.0 {
				t.Errorf("unexpected cost for %s: $%f per 1M tokens", model, info.CostPer1MTokens)
			}
		})
	}
}

func TestEmbeddingModelDimensions(t *testing.T) {
	tests := []struct {
		model           string
		expectedDims    int
	}{
		{"text-embedding-3-small", 1536},
		{"text-embedding-3-large", 3072},
		{"text-embedding-ada-002", 1536},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			info, ok := EmbeddingModels[tt.model]
			if !ok {
				t.Fatalf("model %q not found", tt.model)
			}

			if info.Dimensions != tt.expectedDims {
				t.Errorf("got %d dimensions, want %d", info.Dimensions, tt.expectedDims)
			}
		})
	}
}

func TestEmbeddingRequest(t *testing.T) {
	tests := []struct {
		name  string
		input interface{}
	}{
		{
			name:  "single string",
			input: "Hello, world!",
		},
		{
			name:  "multiple strings",
			input: []string{"Hello", "World"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := EmbeddingRequest{
				Input: tt.input,
				Model: "text-embedding-ada-002",
			}

			if req.Model != "text-embedding-ada-002" {
				t.Errorf("got model %q, want %q", req.Model, "text-embedding-ada-002")
			}

			if req.Input == nil {
				t.Error("input should not be nil")
			}
		})
	}
}

func TestEmbeddingRequestWithDimensions(t *testing.T) {
	dims := 512
	req := EmbeddingRequest{
		Input:      "Test text",
		Model:      "text-embedding-3-small",
		Dimensions: &dims,
	}

	if req.Dimensions == nil {
		t.Fatal("dimensions should not be nil")
	}

	if *req.Dimensions != dims {
		t.Errorf("got dimensions %d, want %d", *req.Dimensions, dims)
	}
}
