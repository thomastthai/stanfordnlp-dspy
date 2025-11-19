package azure

import (
	"context"
	"fmt"
	"os"

	"github.com/stanfordnlp/dspy/internal/clients"
)

// Provider implements the Provider interface for Azure OpenAI.
type Provider struct{}

// supportedModels lists commonly used Azure OpenAI deployment names.
// Note: Azure uses deployment names which can be customized by users.
var supportedModels = []string{
	// GPT-4o series
	"gpt-4o",
	"gpt-4o-mini",

	// GPT-4 Turbo
	"gpt-4-turbo",
	"gpt-4",

	// GPT-3.5 Turbo
	"gpt-35-turbo",
	"gpt-35-turbo-16k",

	// These are base model names - actual deployment names may vary
}

// LM wraps the Azure OpenAI client to implement the BaseLM interface.
type LM struct {
	client *Client
	model  string
}

// NewLM creates a new Azure OpenAI LM client.
func NewLM(model, endpoint, apiKey string) (*LM, error) {
	client, err := NewClient(ClientOptions{
		Endpoint: endpoint,
		APIKey:   apiKey,
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
	return "azure"
}

// Create implements Provider.Create.
func (p *Provider) Create(config map[string]interface{}) (clients.BaseLM, error) {
	// Extract deployment/model name
	model, ok := config["model"].(string)
	if !ok || model == "" {
		return nil, fmt.Errorf("model/deployment name is required")
	}

	// Extract Azure endpoint from config or environment
	endpoint, ok := config["endpoint"].(string)
	if !ok || endpoint == "" {
		endpoint = os.Getenv("AZURE_OPENAI_ENDPOINT")
		if endpoint == "" {
			return nil, fmt.Errorf("Azure OpenAI endpoint not found in config or AZURE_OPENAI_ENDPOINT environment variable")
		}
	}

	// Extract API key from config or environment
	apiKey, ok := config["api_key"].(string)
	if !ok || apiKey == "" {
		apiKey = os.Getenv("AZURE_OPENAI_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("Azure OpenAI API key not found in config or AZURE_OPENAI_API_KEY environment variable")
		}
	}

	return NewLM(model, endpoint, apiKey)
}

// Name implements Provider.Name.
func (p *Provider) Name() string {
	return "azure"
}

// SupportedModels implements Provider.SupportedModels.
func (p *Provider) SupportedModels() []string {
	return supportedModels
}

func init() {
	// Register the Azure OpenAI provider
	clients.RegisterProvider("azure", &Provider{})
}
