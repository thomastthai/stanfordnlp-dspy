package anthropic

import (
	"context"
	"fmt"
	"os"

	"github.com/stanfordnlp/dspy/internal/clients"
)

// Provider implements the Provider interface for Anthropic/Claude.
type Provider struct{}

// supportedModels lists all supported Anthropic Claude models.
var supportedModels = []string{
	// Claude 3 Opus
	"claude-3-opus-20240229",
	"claude-3-opus",
	
	// Claude 3 Sonnet
	"claude-3-sonnet-20240229",
	"claude-3-sonnet",
	"claude-3-5-sonnet-20240620",
	"claude-3-5-sonnet",
	
	// Claude 3 Haiku
	"claude-3-haiku-20240307",
	"claude-3-haiku",
	
	// Claude 2
	"claude-2.1",
	"claude-2.0",
	"claude-2",
	
	// Claude Instant
	"claude-instant-1.2",
	"claude-instant-1",
}

// LM wraps the Anthropic client to implement the BaseLM interface.
type LM struct {
	client *Client
	model  string
}

// NewLM creates a new Anthropic LM client.
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
	return "anthropic"
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
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("Anthropic API key not found in config or ANTHROPIC_API_KEY environment variable")
		}
	}

	return NewLM(model, apiKey)
}

// Name implements Provider.Name.
func (p *Provider) Name() string {
	return "anthropic"
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

func init() {
	// Register the Anthropic provider
	clients.RegisterProvider("anthropic", &Provider{})
}
