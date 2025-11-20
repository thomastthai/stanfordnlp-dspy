package retrievers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Marqo provides retrieval using Marqo multimodal search engine.
type Marqo struct {
	*BaseRetriever
	url       string
	apiKey    string
	indexName string
	client    *http.Client
}

// MarqoOptions configures the Marqo retriever.
type MarqoOptions struct {
	URL       string
	APIKey    string
	IndexName string
	Timeout   time.Duration
}

// DefaultMarqoOptions returns default Marqo options.
func DefaultMarqoOptions() MarqoOptions {
	return MarqoOptions{
		URL:       "http://localhost:8882",
		IndexName: "documents",
		Timeout:   30 * time.Second,
	}
}

// NewMarqo creates a new Marqo retriever.
func NewMarqo(opts MarqoOptions) *Marqo {
	return &Marqo{
		BaseRetriever: NewBaseRetriever("marqo"),
		url:           opts.URL,
		apiKey:        opts.APIKey,
		indexName:     opts.IndexName,
		client: &http.Client{
			Timeout: opts.Timeout,
		},
	}
}

// Retrieve implements Retriever.Retrieve.
func (m *Marqo) Retrieve(ctx context.Context, query string, k int) ([]string, error) {
	docs, err := m.RetrieveWithScores(ctx, query, k)
	if err != nil {
		return nil, err
	}

	results := make([]string, len(docs))
	for i, doc := range docs {
		results[i] = doc.Content
	}

	return results, nil
}

// RetrieveWithScores implements Retriever.RetrieveWithScores.
func (m *Marqo) RetrieveWithScores(ctx context.Context, query string, k int) ([]Document, error) {
	return m.Search(ctx, query, k, nil)
}

// Search performs a search with optional filters.
func (m *Marqo) Search(ctx context.Context, query string, limit int, filter map[string]interface{}) ([]Document, error) {
	payload := map[string]interface{}{
		"q":     query,
		"limit": limit,
	}

	if filter != nil {
		payload["filter"] = filter
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	url := fmt.Sprintf("%s/indexes/%s/search", m.url, m.indexName)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if m.apiKey != "" {
		req.Header.Set("x-api-key", m.apiKey)
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var result struct {
		Hits []struct {
			ID         string                 `json:"_id"`
			Score      float64                `json:"_score"`
			Highlights []interface{}          `json:"_highlights"`
			Document   map[string]interface{} `json:",inline"`
		} `json:"hits"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	docs := make([]Document, 0, len(result.Hits))
	for _, hit := range result.Hits {
		doc := Document{
			ID:       hit.ID,
			Score:    hit.Score,
			Metadata: hit.Document,
		}

		// Extract text content from document fields
		// Common field names for text content
		for _, field := range []string{"text", "content", "description", "body"} {
			if text, ok := hit.Document[field].(string); ok {
				doc.Content = text
				break
			}
		}

		docs = append(docs, doc)
	}

	return docs, nil
}

// AddDocuments adds documents to the index.
func (m *Marqo) AddDocuments(ctx context.Context, documents []map[string]interface{}) error {
	payload := map[string]interface{}{
		"documents": documents,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal documents: %w", err)
	}

	url := fmt.Sprintf("%s/indexes/%s/documents", m.url, m.indexName)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if m.apiKey != "" {
		req.Header.Set("x-api-key", m.apiKey)
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	return nil
}

// CreateIndex creates a new index with the given settings.
func (m *Marqo) CreateIndex(ctx context.Context, settings map[string]interface{}) error {
	payload := settings
	if payload == nil {
		payload = make(map[string]interface{})
	}
	payload["indexName"] = m.indexName

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal settings: %w", err)
	}

	url := fmt.Sprintf("%s/indexes/%s", m.url, m.indexName)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if m.apiKey != "" {
		req.Header.Set("x-api-key", m.apiKey)
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	return nil
}

// MultimodalSearch performs a search with multimodal query (text and/or images).
func (m *Marqo) MultimodalSearch(ctx context.Context, query interface{}, limit int) ([]Document, error) {
	payload := map[string]interface{}{
		"q":     query,
		"limit": limit,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	url := fmt.Sprintf("%s/indexes/%s/search", m.url, m.indexName)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if m.apiKey != "" {
		req.Header.Set("x-api-key", m.apiKey)
	}

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	// Parse response similar to Search
	var result struct {
		Hits []struct {
			ID       string                 `json:"_id"`
			Score    float64                `json:"_score"`
			Document map[string]interface{} `json:",inline"`
		} `json:"hits"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	docs := make([]Document, 0, len(result.Hits))
	for _, hit := range result.Hits {
		doc := Document{
			ID:       hit.ID,
			Score:    hit.Score,
			Metadata: hit.Document,
		}

		// Extract content
		for _, field := range []string{"text", "content", "image_url"} {
			if val, ok := hit.Document[field]; ok {
				if text, ok := val.(string); ok {
					doc.Content = text
					break
				}
			}
		}

		docs = append(docs, doc)
	}

	return docs, nil
}
