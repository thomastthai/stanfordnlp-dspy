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

// CohereEmbedder uses Cohere's embedding API.
type CohereEmbedder struct {
	apiKey      string
	model       string
	inputType   string
	dimension   int
	baseURL     string
	client      *http.Client
	compression string
}

// CohereConfig configures the Cohere embedder.
type CohereConfig struct {
	APIKey      string
	Model       string // "embed-english-v3.0", "embed-multilingual-v3.0"
	InputType   string // "search_document", "search_query", "classification", "clustering"
	BaseURL     string
	Compression string // "none", "ubinary", "binary"
}

// cohereRequest is the request format for Cohere embeddings API.
type cohereRequest struct {
	Texts       []string `json:"texts"`
	Model       string   `json:"model"`
	InputType   string   `json:"input_type,omitempty"`
	Truncate    string   `json:"truncate,omitempty"`
	Compression string   `json:"embedding_types,omitempty"`
}

// cohereResponse is the response format from Cohere embeddings API.
type cohereResponse struct {
	ID         string `json:"id"`
	Embeddings []struct {
		Values []float32 `json:"values,omitempty"`
		Float  []float32 `json:"float,omitempty"`
	} `json:"embeddings"`
	Texts []string `json:"texts"`
	Meta  struct {
		APIVersion struct {
			Version string `json:"version"`
		} `json:"api_version"`
		BilledUnits struct {
			InputTokens int `json:"input_tokens"`
		} `json:"billed_units"`
	} `json:"meta"`
}

// NewCohereEmbedder creates a new Cohere embedder.
func NewCohereEmbedder(config CohereConfig) (*CohereEmbedder, error) {
	if config.APIKey == "" {
		config.APIKey = os.Getenv("COHERE_API_KEY")
	}
	if config.APIKey == "" {
		return nil, fmt.Errorf("Cohere API key is required")
	}

	if config.Model == "" {
		config.Model = "embed-english-v3.0"
	}

	if config.InputType == "" {
		config.InputType = "search_document"
	}

	if config.BaseURL == "" {
		config.BaseURL = "https://api.cohere.ai/v1"
	}

	if config.Compression == "" {
		config.Compression = "none"
	}

	// Determine dimension based on model
	dimension := 1024
	if config.Model == "embed-english-v3.0" || config.Model == "embed-multilingual-v3.0" {
		dimension = 1024
	}

	return &CohereEmbedder{
		apiKey:      config.APIKey,
		model:       config.Model,
		inputType:   config.InputType,
		dimension:   dimension,
		baseURL:     config.BaseURL,
		client:      &http.Client{},
		compression: config.Compression,
	}, nil
}

// Embed implements Embedder.Embed.
func (e *CohereEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	// Prepare request
	reqBody := cohereRequest{
		Texts:     texts,
		Model:     e.model,
		InputType: e.inputType,
		Truncate:  "END",
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	url := fmt.Sprintf("%s/embed", e.baseURL)
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
	var apiResp cohereResponse
	if err := json.Unmarshal(respBody, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Extract embeddings
	embeddings := make([][]float32, len(texts))
	for i, emb := range apiResp.Embeddings {
		if i >= len(embeddings) {
			break
		}
		// Try both possible field names
		if len(emb.Float) > 0 {
			embeddings[i] = emb.Float
		} else {
			embeddings[i] = emb.Values
		}
	}

	return embeddings, nil
}

// Dimension implements Embedder.Dimension.
func (e *CohereEmbedder) Dimension() int {
	return e.dimension
}

// MaxBatchSize implements Embedder.MaxBatchSize.
func (e *CohereEmbedder) MaxBatchSize() int {
	return 96 // Cohere's batch limit
}
