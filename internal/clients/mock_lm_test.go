package clients

import (
	"context"
	"testing"
)

func TestMockLM_Call(t *testing.T) {
	lm := NewMockLM("test-model")

	req := NewRequest().
		WithMessages(NewMessage("user", "Hello")).
		WithTemperature(0.7)

	resp, err := lm.Call(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Choices) != 1 {
		t.Errorf("expected 1 choice, got %d", len(resp.Choices))
	}

	if resp.Model != "test-model" {
		t.Errorf("expected model 'test-model', got %s", resp.Model)
	}
}

func TestMockLM_CustomResponse(t *testing.T) {
	lm := NewMockLM("test-model")
	lm.ResponseFunc = func(req *Request) (*Response, error) {
		return &Response{
			Choices: []Choice{
				{
					Message: Message{
						Role:    "assistant",
						Content: "custom response",
					},
					Text: "custom response",
				},
			},
		}, nil
	}

	req := NewRequest().WithPrompt("test")
	resp, err := lm.Call(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Choices[0].Text != "custom response" {
		t.Errorf("expected 'custom response', got %s", resp.Choices[0].Text)
	}
}

func TestProviderRegistry(t *testing.T) {
	providers := ListProviders()
	found := false
	for _, p := range providers {
		if p == "mock" {
			found = true
			break
		}
	}

	if !found {
		t.Error("mock provider not found in registry")
	}
}

func TestCreateClient(t *testing.T) {
	config := map[string]interface{}{
		"model": "mock-gpt-4",
	}

	client, err := CreateClient("mock", config)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	if client.Name() != "mock-gpt-4" {
		t.Errorf("expected model 'mock-gpt-4', got %s", client.Name())
	}
}
