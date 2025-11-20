package gemini

import (
	"context"
	"fmt"
	"os"

	"github.com/stanfordnlp/dspy/internal/clients"
)

// Provider implements the Provider interface for Google Gemini.
type Provider struct{}

// supportedModels lists all supported Google Gemini models.
var supportedModels = []string{
	// Gemini 1.5 Pro
	"gemini-1.5-pro",
	"gemini-1.5-pro-latest",

	// Gemini 1.5 Flash
	"gemini-1.5-flash",
	"gemini-1.5-flash-latest",

	// Gemini 1.0 Pro
	"gemini-1.0-pro",
	"gemini-pro",

	// Gemini Pro Vision
	"gemini-pro-vision",
	"gemini-1.0-pro-vision",
}

// LM wraps the Gemini client to implement the BaseLM interface.
type LM struct {
	client *Client
	model  string
}

// NewLM creates a new Gemini LM client.
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
	return "gemini"
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
		apiKey = os.Getenv("GOOGLE_API_KEY")
		if apiKey == "" {
			apiKey = os.Getenv("GEMINI_API_KEY")
			if apiKey == "" {
				return nil, fmt.Errorf("Google API key not found in config or GOOGLE_API_KEY/GEMINI_API_KEY environment variable")
			}
		}
	}

	return NewLM(model, apiKey)
}

// Name implements Provider.Name.
func (p *Provider) Name() string {
	return "gemini"
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
	// Register the Gemini provider
	clients.RegisterProvider("gemini", &Provider{})
}
