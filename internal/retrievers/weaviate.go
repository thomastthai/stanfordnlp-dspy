package retrievers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Weaviate provides retrieval using Weaviate vector database.
type Weaviate struct {
	*BaseRetriever
	url             string
	apiKey          string
	collectionName  string
	textKey         string
	client          *http.Client
	headers         map[string]string
}

// WeaviateOptions configures the Weaviate retriever.
type WeaviateOptions struct {
	URL            string
	APIKey         string
	CollectionName string
	TextKey        string
	Timeout        time.Duration
	Headers        map[string]string
}

// DefaultWeaviateOptions returns default Weaviate options.
func DefaultWeaviateOptions() WeaviateOptions {
	return WeaviateOptions{
		URL:            "http://localhost:8080",
		CollectionName: "Documents",
		TextKey:        "content",
		Timeout:        30 * time.Second,
		Headers:        make(map[string]string),
	}
}

// NewWeaviate creates a new Weaviate retriever.
func NewWeaviate(opts WeaviateOptions) *Weaviate {
	return &Weaviate{
		BaseRetriever:  NewBaseRetriever("weaviate"),
		url:            opts.URL,
		apiKey:         opts.APIKey,
		collectionName: opts.CollectionName,
		textKey:        opts.TextKey,
		client: &http.Client{
			Timeout: opts.Timeout,
		},
		headers: opts.Headers,
	}
}

// Retrieve implements Retriever.Retrieve.
func (w *Weaviate) Retrieve(ctx context.Context, query string, k int) ([]string, error) {
	docs, err := w.RetrieveWithScores(ctx, query, k)
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
func (w *Weaviate) RetrieveWithScores(ctx context.Context, query string, k int) ([]Document, error) {
	// Use hybrid search (combines dense and sparse)
	return w.hybridSearch(ctx, query, k)
}

// hybridSearch performs a hybrid search combining vector and keyword search.
func (w *Weaviate) hybridSearch(ctx context.Context, query string, k int) ([]Document, error) {
	// Build GraphQL query
	graphQLQuery := fmt.Sprintf(`{
		Get {
			%s(
				hybrid: {
					query: "%s"
				}
				limit: %d
			) {
				%s
				_additional {
					score
					id
				}
			}
		}
	}`, w.collectionName, w.escapeString(query), k, w.textKey)
	
	return w.executeGraphQL(ctx, graphQLQuery)
}

// VectorSearch performs a pure vector similarity search.
func (w *Weaviate) VectorSearch(ctx context.Context, query string, k int) ([]Document, error) {
	graphQLQuery := fmt.Sprintf(`{
		Get {
			%s(
				nearText: {
					concepts: ["%s"]
				}
				limit: %d
			) {
				%s
				_additional {
					score
					id
					distance
				}
			}
		}
	}`, w.collectionName, w.escapeString(query), k, w.textKey)
	
	return w.executeGraphQL(ctx, graphQLQuery)
}

// executeGraphQL executes a GraphQL query against Weaviate.
func (w *Weaviate) executeGraphQL(ctx context.Context, query string) ([]Document, error) {
	payload := map[string]string{
		"query": query,
	}
	
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal query: %w", err)
	}
	
	url := fmt.Sprintf("%s/v1/graphql", w.url)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	req.Header.Set("Content-Type", "application/json")
	if w.apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", w.apiKey))
	}
	
	// Add custom headers
	for k, v := range w.headers {
		req.Header.Set(k, v)
	}
	
	resp, err := w.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}
	
	return w.parseGraphQLResponse(result)
}

// parseGraphQLResponse parses the Weaviate GraphQL response.
func (w *Weaviate) parseGraphQLResponse(response map[string]interface{}) ([]Document, error) {
	data, ok := response["data"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid response format: missing data field")
	}
	
	get, ok := data["Get"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid response format: missing Get field")
	}
	
	collection, ok := get[w.collectionName].([]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid response format: missing collection field")
	}
	
	docs := make([]Document, 0, len(collection))
	for _, item := range collection {
		obj, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		
		doc := Document{
			Metadata: make(map[string]interface{}),
		}
		
		// Extract text content
		if text, ok := obj[w.textKey].(string); ok {
			doc.Content = text
		}
		
		// Extract additional metadata
		if additional, ok := obj["_additional"].(map[string]interface{}); ok {
			if score, ok := additional["score"].(float64); ok {
				doc.Score = score
			}
			if id, ok := additional["id"].(string); ok {
				doc.ID = id
			}
			if distance, ok := additional["distance"].(float64); ok {
				doc.Metadata["distance"] = distance
			}
		}
		
		// Store all fields as metadata
		for k, v := range obj {
			if k != w.textKey && k != "_additional" {
				doc.Metadata[k] = v
			}
		}
		
		docs = append(docs, doc)
	}
	
	return docs, nil
}

// escapeString escapes special characters for GraphQL.
func (w *Weaviate) escapeString(s string) string {
	// Simple escape - replace quotes
	b := []byte(s)
	b = bytes.ReplaceAll(b, []byte(`"`), []byte(`\"`))
	b = bytes.ReplaceAll(b, []byte("\n"), []byte(`\n`))
	return string(b)
}

// CreateObject creates a new object in Weaviate.
func (w *Weaviate) CreateObject(ctx context.Context, properties map[string]interface{}) (string, error) {
	payload := map[string]interface{}{
		"class":      w.collectionName,
		"properties": properties,
	}
	
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal object: %w", err)
	}
	
	url := fmt.Sprintf("%s/v1/objects", w.url)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	
	req.Header.Set("Content-Type", "application/json")
	if w.apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", w.apiKey))
	}
	
	resp, err := w.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return "", fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}
	
	id, _ := result["id"].(string)
	return id, nil
}
