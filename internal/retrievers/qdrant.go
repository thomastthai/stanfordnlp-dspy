package retrievers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Qdrant provides retrieval using Qdrant vector database.
type Qdrant struct {
	*BaseRetriever
	url            string
	apiKey         string
	collectionName string
	client         *http.Client
}

// QdrantOptions configures the Qdrant retriever.
type QdrantOptions struct {
	URL            string
	APIKey         string
	CollectionName string
	Timeout        time.Duration
}

// DefaultQdrantOptions returns default Qdrant options.
func DefaultQdrantOptions() QdrantOptions {
	return QdrantOptions{
		URL:            "http://localhost:6333",
		CollectionName: "documents",
		Timeout:        30 * time.Second,
	}
}

// NewQdrant creates a new Qdrant retriever.
func NewQdrant(opts QdrantOptions) *Qdrant {
	return &Qdrant{
		BaseRetriever:  NewBaseRetriever("qdrant"),
		url:            opts.URL,
		apiKey:         opts.APIKey,
		collectionName: opts.CollectionName,
		client: &http.Client{
			Timeout: opts.Timeout,
		},
	}
}

// Retrieve implements Retriever.Retrieve.
func (q *Qdrant) Retrieve(ctx context.Context, query string, k int) ([]string, error) {
	docs, err := q.RetrieveWithScores(ctx, query, k)
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
func (q *Qdrant) RetrieveWithScores(ctx context.Context, query string, k int) ([]Document, error) {
	// This is a simplified implementation that would need embedding generation
	// In practice, you'd need to generate embeddings for the query first
	return nil, fmt.Errorf("Qdrant requires embedding generation - use SearchByVector instead")
}

// SearchByVector searches using a pre-computed vector.
func (q *Qdrant) SearchByVector(ctx context.Context, vector []float64, k int, filter map[string]interface{}) ([]Document, error) {
	payload := map[string]interface{}{
		"vector": vector,
		"limit":  k,
	}

	if filter != nil {
		payload["filter"] = filter
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	url := fmt.Sprintf("%s/collections/%s/points/search", q.url, q.collectionName)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if q.apiKey != "" {
		req.Header.Set("api-key", q.apiKey)
	}

	resp, err := q.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var result struct {
		Result []struct {
			ID      interface{}            `json:"id"`
			Score   float64                `json:"score"`
			Payload map[string]interface{} `json:"payload"`
		} `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	docs := make([]Document, 0, len(result.Result))
	for _, item := range result.Result {
		doc := Document{
			Score:    item.Score,
			Metadata: item.Payload,
		}

		// Extract ID
		if id, ok := item.ID.(string); ok {
			doc.ID = id
		} else if id, ok := item.ID.(float64); ok {
			doc.ID = fmt.Sprintf("%d", int(id))
		}

		// Extract content from payload
		if text, ok := item.Payload["text"].(string); ok {
			doc.Content = text
		} else if content, ok := item.Payload["content"].(string); ok {
			doc.Content = content
		}

		docs = append(docs, doc)
	}

	return docs, nil
}

// UpsertPoints upserts points (documents) into the collection.
func (q *Qdrant) UpsertPoints(ctx context.Context, points []QdrantPoint) error {
	payload := map[string]interface{}{
		"points": points,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal points: %w", err)
	}

	url := fmt.Sprintf("%s/collections/%s/points", q.url, q.collectionName)
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if q.apiKey != "" {
		req.Header.Set("api-key", q.apiKey)
	}

	resp, err := q.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	return nil
}

// QdrantPoint represents a point in Qdrant.
type QdrantPoint struct {
	ID      interface{}            `json:"id"`
	Vector  []float64              `json:"vector"`
	Payload map[string]interface{} `json:"payload"`
}

// CreateCollection creates a new collection in Qdrant.
func (q *Qdrant) CreateCollection(ctx context.Context, vectorSize int, distance string) error {
	payload := map[string]interface{}{
		"vectors": map[string]interface{}{
			"size":     vectorSize,
			"distance": distance, // "Cosine", "Euclid", or "Dot"
		},
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal payload: %w", err)
	}

	url := fmt.Sprintf("%s/collections/%s", q.url, q.collectionName)
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if q.apiKey != "" {
		req.Header.Set("api-key", q.apiKey)
	}

	resp, err := q.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	return nil
}
