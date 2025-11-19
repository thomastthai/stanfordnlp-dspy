package embeddings

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

// OpenAIEmbedder uses OpenAI's embedding API.
type OpenAIEmbedder struct {
	apiKey    string
	model     string
	dimension int
	baseURL   string
	client    *http.Client
}

// OpenAIConfig configures the OpenAI embedder.
type OpenAIConfig struct {
	APIKey    string
	Model     string // "text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"
	Dimension int    // Optional: for text-embedding-3 models
	BaseURL   string // Optional: custom API endpoint
}

// openAIRequest is the request format for OpenAI embeddings API.
type openAIRequest struct {
	Input      interface{} `json:"input"` // string or []string
	Model      string      `json:"model"`
	Dimensions int         `json:"dimensions,omitempty"`
}

// openAIResponse is the response format from OpenAI embeddings API.
type openAIResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// NewOpenAIEmbedder creates a new OpenAI embedder.
func NewOpenAIEmbedder(config OpenAIConfig) (*OpenAIEmbedder, error) {
	if config.APIKey == "" {
		config.APIKey = os.Getenv("OPENAI_API_KEY")
	}
	if config.APIKey == "" {
		return nil, fmt.Errorf("OpenAI API key is required")
	}
	
	if config.Model == "" {
		config.Model = "text-embedding-3-small"
	}
	
	if config.BaseURL == "" {
		config.BaseURL = "https://api.openai.com/v1"
	}
	
	// Set default dimensions based on model
	if config.Dimension == 0 {
		switch config.Model {
		case "text-embedding-ada-002":
			config.Dimension = 1536
		case "text-embedding-3-small":
			config.Dimension = 1536
		case "text-embedding-3-large":
			config.Dimension = 3072
		default:
			config.Dimension = 1536
		}
	}
	
	return &OpenAIEmbedder{
		apiKey:    config.APIKey,
		model:     config.Model,
		dimension: config.Dimension,
		baseURL:   config.BaseURL,
		client:    &http.Client{},
	}, nil
}

// Embed implements Embedder.Embed.
func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	
	// Prepare request
	reqBody := openAIRequest{
		Input: texts,
		Model: e.model,
	}
	
	// Only set dimensions for text-embedding-3 models
	if e.model == "text-embedding-3-small" || e.model == "text-embedding-3-large" {
		reqBody.Dimensions = e.dimension
	}
	
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	
	// Create HTTP request
	url := fmt.Sprintf("%s/embeddings", e.baseURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", e.apiKey))
	
	// Send request
	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()
	
	// Read response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(respBody))
	}
	
	// Parse response
	var apiResp openAIResponse
	if err := json.Unmarshal(respBody, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	
	// Extract embeddings in correct order
	embeddings := make([][]float32, len(texts))
	for _, data := range apiResp.Data {
		if data.Index < len(embeddings) {
			embeddings[data.Index] = data.Embedding
		}
	}
	
	return embeddings, nil
}

// Dimension implements Embedder.Dimension.
func (e *OpenAIEmbedder) Dimension() int {
	return e.dimension
}

// MaxBatchSize implements Embedder.MaxBatchSize.
func (e *OpenAIEmbedder) MaxBatchSize() int {
	return 2048 // OpenAI's batch limit
}
