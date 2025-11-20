package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/hashicorp/go-retryablehttp"
)

// EmbeddingRequest represents an OpenAI embeddings API request.
type EmbeddingRequest struct {
	Input          interface{} `json:"input"` // string or []string
	Model          string      `json:"model"`
	EncodingFormat string      `json:"encoding_format,omitempty"` // "float" or "base64"
	Dimensions     *int        `json:"dimensions,omitempty"`      // For text-embedding-3 models
	User           string      `json:"user,omitempty"`
}

// EmbeddingResponse represents an OpenAI embeddings API response.
type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  EmbeddingUsage  `json:"usage"`
}

// EmbeddingData represents a single embedding.
type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// EmbeddingUsage represents token usage for embeddings.
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// CreateEmbeddings generates embeddings for the given texts.
func (c *Client) CreateEmbeddings(ctx context.Context, texts []string, model string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("no texts provided")
	}

	if model == "" {
		model = "text-embedding-ada-002"
	}

	// Build request
	var input interface{}
	if len(texts) == 1 {
		input = texts[0]
	} else {
		input = texts
	}

	req := EmbeddingRequest{
		Input: input,
		Model: model,
	}

	// Marshal to JSON
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := retryablehttp.NewRequestWithContext(ctx, "POST", c.baseURL+"/embeddings", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	if c.orgID != "" {
		httpReq.Header.Set("OpenAI-Organization", c.orgID)
	}

	// Make API call
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		var errResp ErrorResponse
		if err := json.Unmarshal(body, &errResp); err != nil {
			return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
		}
		return nil, fmt.Errorf("API error: %s (type: %s, code: %v)", errResp.Error.Message, errResp.Error.Type, errResp.Error.Code)
	}

	// Parse response
	var embeddingResp EmbeddingResponse
	if err := json.Unmarshal(body, &embeddingResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Extract embeddings
	embeddings := make([][]float32, len(embeddingResp.Data))
	for _, data := range embeddingResp.Data {
		embeddings[data.Index] = data.Embedding
	}

	return embeddings, nil
}

// CreateEmbedding generates a single embedding for the given text.
func (c *Client) CreateEmbedding(ctx context.Context, text, model string) ([]float32, error) {
	embeddings, err := c.CreateEmbeddings(ctx, []string{text}, model)
	if err != nil {
		return nil, err
	}

	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return embeddings[0], nil
}

// CreateEmbeddingsWithDimensions generates embeddings with custom dimensions (for text-embedding-3 models).
func (c *Client) CreateEmbeddingsWithDimensions(ctx context.Context, texts []string, model string, dimensions int) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("no texts provided")
	}

	if model == "" {
		model = "text-embedding-3-small"
	}

	// Build request
	var input interface{}
	if len(texts) == 1 {
		input = texts[0]
	} else {
		input = texts
	}

	req := EmbeddingRequest{
		Input:      input,
		Model:      model,
		Dimensions: &dimensions,
	}

	// Marshal to JSON
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := retryablehttp.NewRequestWithContext(ctx, "POST", c.baseURL+"/embeddings", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	if c.orgID != "" {
		httpReq.Header.Set("OpenAI-Organization", c.orgID)
	}

	// Make API call
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		var errResp ErrorResponse
		if err := json.Unmarshal(body, &errResp); err != nil {
			return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
		}
		return nil, fmt.Errorf("API error: %s (type: %s)", errResp.Error.Message, errResp.Error.Type)
	}

	// Parse response
	var embeddingResp EmbeddingResponse
	if err := json.Unmarshal(body, &embeddingResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Extract embeddings
	embeddings := make([][]float32, len(embeddingResp.Data))
	for _, data := range embeddingResp.Data {
		embeddings[data.Index] = data.Embedding
	}

	return embeddings, nil
}

// EmbeddingModels contains information about available embedding models.
var EmbeddingModels = map[string]struct {
	Name            string
	Dimensions      int
	MaxTokens       int
	CostPer1MTokens float64
}{
	"text-embedding-3-small": {
		Name:            "text-embedding-3-small",
		Dimensions:      1536,
		MaxTokens:       8191,
		CostPer1MTokens: 0.02,
	},
	"text-embedding-3-large": {
		Name:            "text-embedding-3-large",
		Dimensions:      3072,
		MaxTokens:       8191,
		CostPer1MTokens: 0.13,
	},
	"text-embedding-ada-002": {
		Name:            "text-embedding-ada-002",
		Dimensions:      1536,
		MaxTokens:       8191,
		CostPer1MTokens: 0.10,
	},
}

// GetEmbeddingModelInfo returns information about an embedding model.
func GetEmbeddingModelInfo(model string) (struct {
	Name            string
	Dimensions      int
	MaxTokens       int
	CostPer1MTokens float64
}, bool) {
	info, ok := EmbeddingModels[model]
	return info, ok
}
