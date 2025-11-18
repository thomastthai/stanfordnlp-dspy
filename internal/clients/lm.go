// Package clients provides a unified LM client with provider routing.
package clients

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// LM is the main language model client with automatic provider routing.
type LM struct {
	provider     string
	model        string
	client       BaseLM
	cache        Cache
	usageTracker *UsageTracker
}

// Cache defines the interface for caching LM responses.
type Cache interface {
	Get(ctx context.Context, key string) ([]byte, bool, error)
	Set(ctx context.Context, key string, value []byte, ttl time.Duration) error
	Delete(ctx context.Context, key string) error
	Clear(ctx context.Context) error
}

// UsageTracker tracks token usage and costs.
type UsageTracker struct {
	TotalPromptTokens     int
	TotalCompletionTokens int
	TotalTokens           int
	TotalCost             float64
	RequestCount          int
}

// LMOptions configures the LM client.
type LMOptions struct {
	// Model is the model identifier (e.g., "openai/gpt-4o-mini", "gpt-4o-mini")
	Model string

	// APIKey is the API key for the provider
	APIKey string

	// Cache is the cache implementation
	Cache Cache

	// CacheTTL is the time-to-live for cached responses
	CacheTTL time.Duration

	// Additional provider-specific config
	Config map[string]interface{}
}

// NewLM creates a new LM client with automatic provider routing.
func NewLM(opts LMOptions) (*LM, error) {
	if opts.Model == "" {
		return nil, fmt.Errorf("model is required")
	}

	// Parse model string to determine provider and model name
	provider, model := parseModelString(opts.Model)

	// Get or create provider
	providerObj, err := GetProvider(provider)
	if err != nil {
		return nil, fmt.Errorf("failed to get provider '%s': %w", provider, err)
	}

	// Create client config
	config := make(map[string]interface{})
	if opts.Config != nil {
		for k, v := range opts.Config {
			config[k] = v
		}
	}
	config["model"] = model
	if opts.APIKey != "" {
		config["api_key"] = opts.APIKey
	}

	// Create provider-specific client
	client, err := providerObj.Create(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create client: %w", err)
	}

	lm := &LM{
		provider:     provider,
		model:        model,
		client:       client,
		cache:        opts.Cache,
		usageTracker: &UsageTracker{},
	}

	return lm, nil
}

// parseModelString parses a model string like "openai/gpt-4o-mini" or "gpt-4o-mini".
// Returns (provider, model).
func parseModelString(modelStr string) (string, string) {
	// Check if model string contains provider prefix
	parts := strings.SplitN(modelStr, "/", 2)
	if len(parts) == 2 {
		return parts[0], parts[1]
	}

	// No provider prefix, try to infer from model name
	provider := inferProvider(modelStr)
	return provider, modelStr
}

// inferProvider infers the provider from the model name.
func inferProvider(model string) string {
	model = strings.ToLower(model)

	// OpenAI models
	if strings.HasPrefix(model, "gpt-") || strings.HasPrefix(model, "o1") || strings.HasPrefix(model, "o3") {
		return "openai"
	}

	// Anthropic models
	if strings.HasPrefix(model, "claude-") {
		return "anthropic"
	}

	// Default to openai
	return "openai"
}

// Call sends a request to the LM.
func (lm *LM) Call(ctx context.Context, request *Request) (*Response, error) {
	// Check cache if enabled
	if lm.cache != nil {
		cacheKey, err := generateCacheKey(request, lm.model)
		if err == nil {
			if cachedData, found, err := lm.cache.Get(ctx, cacheKey); err == nil && found {
				// Deserialize cached response
				response, err := deserializeResponse(cachedData)
				if err == nil {
					return response, nil
				}
			}
		}
	}

	// Make the actual API call
	response, err := lm.client.Call(ctx, request)
	if err != nil {
		return nil, err
	}

	// Track usage
	lm.usageTracker.TotalPromptTokens += response.Usage.PromptTokens
	lm.usageTracker.TotalCompletionTokens += response.Usage.CompletionTokens
	lm.usageTracker.TotalTokens += response.Usage.TotalTokens
	lm.usageTracker.RequestCount++

	// Cache the response if enabled
	if lm.cache != nil {
		cacheKey, err := generateCacheKey(request, lm.model)
		if err == nil {
			if cachedData, err := serializeResponse(response); err == nil {
				// Use a default TTL of 1 hour if not specified
				ttl := 1 * time.Hour
				_ = lm.cache.Set(ctx, cacheKey, cachedData, ttl)
			}
		}
	}

	return response, nil
}

// CallBatch sends multiple requests in batch.
func (lm *LM) CallBatch(ctx context.Context, requests []*Request) ([]*Response, error) {
	return lm.client.CallBatch(ctx, requests)
}

// Name returns the model name.
func (lm *LM) Name() string {
	return lm.model
}

// Provider returns the provider name.
func (lm *LM) Provider() string {
	return lm.provider
}

// Usage returns the usage statistics.
func (lm *LM) Usage() UsageTracker {
	return *lm.usageTracker
}

// ResetUsage resets the usage statistics.
func (lm *LM) ResetUsage() {
	lm.usageTracker = &UsageTracker{}
}

// generateCacheKey generates a cache key for a request.
func generateCacheKey(request *Request, model string) (string, error) {
	// Create a cache key structure
	keyData := struct {
		Model       string
		Messages    []Message
		Temperature float64
		MaxTokens   int
		Prompt      string
	}{
		Model:       model,
		Messages:    request.Messages,
		Temperature: request.Temperature,
		MaxTokens:   request.MaxTokens,
		Prompt:      request.Prompt,
	}

	// Use the cache's key generation function
	// This is a simplified version - in production you'd want a more robust implementation
	return fmt.Sprintf("%v", keyData), nil
}

// serializeResponse serializes a response for caching.
func serializeResponse(response *Response) ([]byte, error) {
	// This is a placeholder - in production you'd use proper serialization
	return []byte(fmt.Sprintf("%v", response)), nil
}

// deserializeResponse deserializes a cached response.
func deserializeResponse(data []byte) (*Response, error) {
	// This is a placeholder - in production you'd use proper deserialization
	return nil, fmt.Errorf("deserialization not implemented")
}
