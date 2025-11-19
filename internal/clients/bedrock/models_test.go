package bedrock

import (
	"testing"

	"github.com/stanfordnlp/dspy/internal/clients"
)

func TestGetModelAdapter(t *testing.T) {
	tests := []struct {
		name     string
		modelID  string
		wantType string
	}{
		{
			name:     "anthropic model",
			modelID:  "anthropic.claude-3-opus-20240229-v1:0",
			wantType: "*bedrock.AnthropicAdapter",
		},
		{
			name:     "titan model",
			modelID:  "amazon.titan-text-express-v1",
			wantType: "*bedrock.TitanAdapter",
		},
		{
			name:     "llama model",
			modelID:  "meta.llama2-70b-chat-v1",
			wantType: "*bedrock.LlamaAdapter",
		},
		{
			name:     "ai21 model",
			modelID:  "ai21.j2-ultra-v1",
			wantType: "*bedrock.AI21Adapter",
		},
		{
			name:     "cohere model",
			modelID:  "cohere.command-text-v14",
			wantType: "*bedrock.CohereAdapter",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			adapter := GetModelAdapter(tt.modelID)
			if adapter == nil {
				t.Error("expected adapter, got nil")
			}
		})
	}
}

func TestAnthropicAdapter_BuildRequest(t *testing.T) {
	adapter := &AnthropicAdapter{}

	request := &clients.Request{
		Messages: []clients.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens:   100,
		Temperature: 0.7,
	}

	body, err := adapter.BuildRequest(request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(body) == 0 {
		t.Error("expected non-empty request body")
	}
}

func TestTitanAdapter_BuildRequest(t *testing.T) {
	adapter := &TitanAdapter{}

	request := &clients.Request{
		Prompt:      "Hello, world!",
		MaxTokens:   100,
		Temperature: 0.5,
	}

	body, err := adapter.BuildRequest(request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(body) == 0 {
		t.Error("expected non-empty request body")
	}
}

func TestLlamaAdapter_BuildRequest(t *testing.T) {
	adapter := &LlamaAdapter{}

	request := &clients.Request{
		Messages: []clients.Message{
			{Role: "user", Content: "Tell me a story"},
		},
		MaxTokens: 200,
	}

	body, err := adapter.BuildRequest(request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(body) == 0 {
		t.Error("expected non-empty request body")
	}
}

func TestAI21Adapter_BuildRequest(t *testing.T) {
	adapter := &AI21Adapter{}

	request := &clients.Request{
		Prompt:    "Write a poem",
		MaxTokens: 150,
	}

	body, err := adapter.BuildRequest(request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(body) == 0 {
		t.Error("expected non-empty request body")
	}
}

func TestCohereAdapter_BuildRequest(t *testing.T) {
	adapter := &CohereAdapter{}

	request := &clients.Request{
		Prompt:      "Explain quantum physics",
		MaxTokens:   300,
		Temperature: 0.8,
	}

	body, err := adapter.BuildRequest(request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(body) == 0 {
		t.Error("expected non-empty request body")
	}
}
