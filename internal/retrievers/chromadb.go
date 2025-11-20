package retrievers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// ChromaDB provides retrieval using ChromaDB vector database.
type ChromaDB struct {
	*BaseRetriever
	url            string
	apiKey         string
	collectionName string
	client         *http.Client
}

// ChromaDBOptions configures the ChromaDB retriever.
type ChromaDBOptions struct {
	URL            string
	APIKey         string
	CollectionName string
	Timeout        time.Duration
}

// DefaultChromaDBOptions returns default ChromaDB options.
func DefaultChromaDBOptions() ChromaDBOptions {
	return ChromaDBOptions{
		URL:            "http://localhost:8000",
		CollectionName: "documents",
		Timeout:        30 * time.Second,
	}
}

// NewChromaDB creates a new ChromaDB retriever.
func NewChromaDB(opts ChromaDBOptions) *ChromaDB {
	return &ChromaDB{
		BaseRetriever:  NewBaseRetriever("chromadb"),
		url:            opts.URL,
		apiKey:         opts.APIKey,
		collectionName: opts.CollectionName,
		client: &http.Client{
			Timeout: opts.Timeout,
		},
	}
}

// Retrieve implements Retriever.Retrieve.
func (c *ChromaDB) Retrieve(ctx context.Context, query string, k int) ([]string, error) {
	docs, err := c.RetrieveWithScores(ctx, query, k)
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
func (c *ChromaDB) RetrieveWithScores(ctx context.Context, query string, k int) ([]Document, error) {
	return c.QueryByText(ctx, query, k, nil)
}

// QueryByText queries the collection using text.
func (c *ChromaDB) QueryByText(ctx context.Context, queryText string, nResults int, where map[string]interface{}) ([]Document, error) {
	payload := map[string]interface{}{
		"query_texts": []string{queryText},
		"n_results":   nResults,
	}

	if where != nil {
		payload["where"] = where
	}

	return c.executeQuery(ctx, payload)
}

// QueryByEmbedding queries the collection using embeddings.
func (c *ChromaDB) QueryByEmbedding(ctx context.Context, queryEmbedding []float64, nResults int, where map[string]interface{}) ([]Document, error) {
	payload := map[string]interface{}{
		"query_embeddings": [][]float64{queryEmbedding},
		"n_results":        nResults,
	}

	if where != nil {
		payload["where"] = where
	}

	return c.executeQuery(ctx, payload)
}

// executeQuery executes a query against ChromaDB.
func (c *ChromaDB) executeQuery(ctx context.Context, payload map[string]interface{}) ([]Document, error) {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}

	url := fmt.Sprintf("%s/api/v1/collections/%s/query", c.url, c.collectionName)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var result struct {
		IDs       [][]string                 `json:"ids"`
		Distances [][]float64                `json:"distances"`
		Documents [][]string                 `json:"documents"`
		Metadatas [][]map[string]interface{} `json:"metadatas"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// ChromaDB returns nested arrays (one per query)
	// We sent one query, so take the first result
	if len(result.IDs) == 0 || len(result.IDs[0]) == 0 {
		return []Document{}, nil
	}

	docs := make([]Document, len(result.IDs[0]))
	for i := range result.IDs[0] {
		doc := Document{
			ID:      result.IDs[0][i],
			Content: result.Documents[0][i],
		}

		// Distance is converted to score (lower distance = higher score)
		if i < len(result.Distances[0]) {
			doc.Score = 1.0 / (1.0 + result.Distances[0][i])
		}

		// Add metadata
		if i < len(result.Metadatas[0]) {
			doc.Metadata = result.Metadatas[0][i]
		}

		docs[i] = doc
	}

	return docs, nil
}

// Add adds documents to the collection.
func (c *ChromaDB) Add(ctx context.Context, ids []string, documents []string, metadatas []map[string]interface{}, embeddings [][]float64) error {
	payload := map[string]interface{}{
		"ids":       ids,
		"documents": documents,
	}

	if metadatas != nil {
		payload["metadatas"] = metadatas
	}

	if embeddings != nil {
		payload["embeddings"] = embeddings
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal payload: %w", err)
	}

	url := fmt.Sprintf("%s/api/v1/collections/%s/add", c.url, c.collectionName)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	return nil
}

// CreateCollection creates a new collection in ChromaDB.
func (c *ChromaDB) CreateCollection(ctx context.Context, metadata map[string]interface{}) error {
	payload := map[string]interface{}{
		"name": c.collectionName,
	}

	if metadata != nil {
		payload["metadata"] = metadata
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal payload: %w", err)
	}

	url := fmt.Sprintf("%s/api/v1/collections", c.url)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiKey))
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	return nil
}
