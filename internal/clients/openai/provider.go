package openai

import (
	"context"
	"fmt"
	"os"

	"github.com/stanfordnlp/dspy/internal/clients"
)

// Provider implements the Provider interface for OpenAI.
type Provider struct{}

// supportedModels lists all supported OpenAI models.
var supportedModels = []string{
	// GPT-4o series
	"gpt-4o",
	"gpt-4o-2024-11-20",
	"gpt-4o-2024-08-06",
	"gpt-4o-2024-05-13",
	"gpt-4o-mini",
	"gpt-4o-mini-2024-07-18",

	// GPT-4 Turbo
	"gpt-4-turbo",
	"gpt-4-turbo-2024-04-09",
	"gpt-4-turbo-preview",
	"gpt-4-0125-preview",
	"gpt-4-1106-preview",

	// GPT-4
	"gpt-4",
	"gpt-4-0613",
	"gpt-4-0314",

	// GPT-3.5 Turbo
	"gpt-3.5-turbo",
	"gpt-3.5-turbo-0125",
	"gpt-3.5-turbo-1106",
	"gpt-3.5-turbo-0613",

	// O1 series (reasoning models)
	"o1",
	"o1-preview",
	"o1-mini",
	"o1-2024-12-17",

	// O3 series (reasoning models)
	"o3",
	"o3-mini",
}

// LM wraps the OpenAI client to implement the BaseLM interface.
type LM struct {
	client *Client
	model  string
}

// NewLM creates a new OpenAI LM client.
func NewLM(model, apiKey string) (*LM, error) {
	client, err := NewClient(ClientOptions{
		APIKey: apiKey,
	})
	if err != nil {
		return nil, err
	}

	return &LM{
		client: client,
		model:  model,
	}, nil
}

// Call implements BaseLM.Call.
func (lm *LM) Call(ctx context.Context, request *clients.Request) (*clients.Response, error) {
	return lm.client.Call(ctx, request, lm.model)
}

// CallBatch implements BaseLM.CallBatch.
func (lm *LM) CallBatch(ctx context.Context, requests []*clients.Request) ([]*clients.Response, error) {
	responses := make([]*clients.Response, len(requests))
	for i, req := range requests {
		resp, err := lm.Call(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("batch request %d failed: %w", i, err)
		}
		responses[i] = resp
	}
	return responses, nil
}

// Name implements BaseLM.Name.
func (lm *LM) Name() string {
	return lm.model
}

// Provider implements BaseLM.Provider.
func (lm *LM) Provider() string {
	return "openai"
}

// Create implements Provider.Create.
func (p *Provider) Create(config map[string]interface{}) (clients.BaseLM, error) {
	// Extract model name
	model, ok := config["model"].(string)
	if !ok || model == "" {
		return nil, fmt.Errorf("model name is required")
	}

	// Validate model
	if !isModelSupported(model) {
		return nil, fmt.Errorf("unsupported model: %s", model)
	}

	// Extract API key from config or environment
	apiKey, ok := config["api_key"].(string)
	if !ok || apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("OpenAI API key not found in config or OPENAI_API_KEY environment variable")
		}
	}

	return NewLM(model, apiKey)
}

// Name implements Provider.Name.
func (p *Provider) Name() string {
	return "openai"
}

// SupportedModels implements Provider.SupportedModels.
func (p *Provider) SupportedModels() []string {
	return supportedModels
}

// isModelSupported checks if a model is supported.
func isModelSupported(model string) bool {
	for _, m := range supportedModels {
		if m == model {
			return true
		}
	}
	return false
}

// IsReasoningModel returns true if the model is an o1 or o3 reasoning model.
func IsReasoningModel(model string) bool {
	return len(model) >= 2 && (model[:2] == "o1" || model[:2] == "o3")
}

func init() {
	// Register the OpenAI provider
	clients.RegisterProvider("openai", &Provider{})
}
