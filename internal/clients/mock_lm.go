package clients

import (
	"context"
	"fmt"
)

// MockLM is a mock language model client for testing.
type MockLM struct {
	name     string
	provider string

	// ResponseFunc can be set to customize the response behavior
	ResponseFunc func(*Request) (*Response, error)
}

// NewMockLM creates a new mock LM client.
func NewMockLM(name string) *MockLM {
	return &MockLM{
		name:     name,
		provider: "mock",
	}
}

// Call implements BaseLM.Call.
func (m *MockLM) Call(ctx context.Context, request *Request) (*Response, error) {
	if m.ResponseFunc != nil {
		return m.ResponseFunc(request)
	}

	// Default mock behavior
	var content string
	if len(request.Messages) > 0 {
		lastMsg := request.Messages[len(request.Messages)-1]
		content = fmt.Sprintf("[Mock response to: %s]", lastMsg.Content)
	} else if request.Prompt != "" {
		content = fmt.Sprintf("[Mock response to: %s]", request.Prompt)
	} else {
		content = "[Mock response]"
	}

	return &Response{
		Choices: []Choice{
			{
				Message: Message{
					Role:    "assistant",
					Content: content,
				},
				Text:         content,
				Index:        0,
				FinishReason: "stop",
			},
		},
		Usage: Usage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
		Model:    m.name,
		ID:       "mock-response-id",
		Metadata: make(map[string]interface{}),
	}, nil
}

// CallBatch implements BaseLM.CallBatch.
func (m *MockLM) CallBatch(ctx context.Context, requests []*Request) ([]*Response, error) {
	responses := make([]*Response, len(requests))
	for i, req := range requests {
		resp, err := m.Call(ctx, req)
		if err != nil {
			return nil, err
		}
		responses[i] = resp
	}
	return responses, nil
}

// Name implements BaseLM.Name.
func (m *MockLM) Name() string {
	return m.name
}

// Provider implements BaseLM.Provider.
func (m *MockLM) Provider() string {
	return m.provider
}

// MockProvider is a mock provider for testing.
type MockProvider struct{}

// Create implements Provider.Create.
func (p *MockProvider) Create(config map[string]interface{}) (BaseLM, error) {
	name := "mock-model"
	if modelName, ok := config["model"].(string); ok {
		name = modelName
	}
	return NewMockLM(name), nil
}

// Name implements Provider.Name.
func (p *MockProvider) Name() string {
	return "mock"
}

// SupportedModels implements Provider.SupportedModels.
func (p *MockProvider) SupportedModels() []string {
	return []string{"mock-model", "mock-gpt-4"}
}

func init() {
	RegisterProvider("mock", &MockProvider{})
}
