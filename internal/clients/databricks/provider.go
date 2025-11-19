package databricks

import (
	"context"
	"fmt"
	"os"

	"github.com/stanfordnlp/dspy/internal/clients"
)

// Provider implements the Provider interface for Databricks.
type Provider struct{}

// supportedModels lists commonly used Databricks foundation models.
var supportedModels = []string{
	// Databricks foundation models
	"databricks-dbrx-instruct",
	"databricks-meta-llama-3-70b-instruct",
	"databricks-meta-llama-3-8b-instruct",
	"databricks-mixtral-8x7b-instruct",
	
	// Custom models - users can specify their own endpoint names
}

// LM wraps the Databricks client to implement the BaseLM interface.
type LM struct {
	client *Client
	model  string
}

// NewLM creates a new Databricks LM client.
func NewLM(model, host, token string) (*LM, error) {
	client, err := NewClient(ClientOptions{
		Host:  host,
		Token: token,
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
	return "databricks"
}

// Create implements Provider.Create.
func (p *Provider) Create(config map[string]interface{}) (clients.BaseLM, error) {
	// Extract model/endpoint name
	model, ok := config["model"].(string)
	if !ok || model == "" {
		return nil, fmt.Errorf("model/endpoint name is required")
	}

	// Extract Databricks host from config or environment
	host, ok := config["host"].(string)
	if !ok || host == "" {
		host = os.Getenv("DATABRICKS_HOST")
		if host == "" {
			return nil, fmt.Errorf("Databricks host not found in config or DATABRICKS_HOST environment variable")
		}
	}

	// Extract token from config or environment
	token, ok := config["token"].(string)
	if !ok || token == "" {
		token = os.Getenv("DATABRICKS_TOKEN")
		if token == "" {
			return nil, fmt.Errorf("Databricks token not found in config or DATABRICKS_TOKEN environment variable")
		}
	}

	return NewLM(model, host, token)
}

// Name implements Provider.Name.
func (p *Provider) Name() string {
	return "databricks"
}

// SupportedModels implements Provider.SupportedModels.
func (p *Provider) SupportedModels() []string {
	return supportedModels
}

func init() {
	// Register the Databricks provider
	clients.RegisterProvider("databricks", &Provider{})
}
