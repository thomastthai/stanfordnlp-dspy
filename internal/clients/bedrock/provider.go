package bedrock

import (
	"context"
	"fmt"

	"github.com/stanfordnlp/dspy/internal/clients"
)

// Provider implements the Provider interface for AWS Bedrock.
type Provider struct{}

// supportedModels lists all supported AWS Bedrock models.
var supportedModels = []string{
	// Anthropic Claude on Bedrock
	"anthropic.claude-3-opus-20240229-v1:0",
	"anthropic.claude-3-sonnet-20240229-v1:0",
	"anthropic.claude-3-5-sonnet-20240620-v1:0",
	"anthropic.claude-3-haiku-20240307-v1:0",
	"anthropic.claude-v2:1",
	"anthropic.claude-v2",
	"anthropic.claude-instant-v1",
	
	// Amazon Titan
	"amazon.titan-text-express-v1",
	"amazon.titan-text-lite-v1",
	"amazon.titan-text-premier-v1:0",
	
	// Meta Llama
	"meta.llama2-13b-chat-v1",
	"meta.llama2-70b-chat-v1",
	"meta.llama3-8b-instruct-v1:0",
	"meta.llama3-70b-instruct-v1:0",
	
	// AI21 Labs
	"ai21.j2-ultra-v1",
	"ai21.j2-mid-v1",
	
	// Cohere
	"cohere.command-text-v14",
	"cohere.command-light-text-v14",
}

// LM wraps the Bedrock client to implement the BaseLM interface.
type LM struct {
	client *Client
	model  string
}

// NewLM creates a new Bedrock LM client.
func NewLM(model, region string) (*LM, error) {
	client, err := NewClient(ClientOptions{
		Region: region,
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
	return "bedrock"
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

	// Extract region (default to us-east-1)
	region, ok := config["region"].(string)
	if !ok || region == "" {
		region = "us-east-1"
	}

	return NewLM(model, region)
}

// Name implements Provider.Name.
func (p *Provider) Name() string {
	return "bedrock"
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
	// Register the Bedrock provider
	clients.RegisterProvider("bedrock", &Provider{})
}
