package retrievers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"time"
)

// ColBERTv2 provides retrieval using ColBERTv2 API.
type ColBERTv2 struct {
	*BaseRetriever
	url         string
	usePost     bool
	client      *http.Client
	cache       map[string][]Document
	cacheMutex  sync.RWMutex
	maxRetries  int
	retryDelay  time.Duration
}

// ColBERTv2Options configures the ColBERTv2 retriever.
type ColBERTv2Options struct {
	URL        string
	Port       string
	UsePost    bool
	Timeout    time.Duration
	MaxRetries int
	RetryDelay time.Duration
	CacheSize  int
}

// DefaultColBERTv2Options returns default ColBERTv2 options.
func DefaultColBERTv2Options() ColBERTv2Options {
	return ColBERTv2Options{
		URL:        "http://0.0.0.0",
		Port:       "",
		UsePost:    false,
		Timeout:    10 * time.Second,
		MaxRetries: 3,
		RetryDelay: 1 * time.Second,
		CacheSize:  1000,
	}
}

// NewColBERTv2 creates a new ColBERTv2 retriever.
func NewColBERTv2(opts ColBERTv2Options) *ColBERTv2 {
	url := opts.URL
	if opts.Port != "" {
		url = fmt.Sprintf("%s:%s", opts.URL, opts.Port)
	}
	
	return &ColBERTv2{
		BaseRetriever: NewBaseRetriever("colbertv2"),
		url:           url,
		usePost:       opts.UsePost,
		client: &http.Client{
			Timeout: opts.Timeout,
		},
		cache:      make(map[string][]Document),
		maxRetries: opts.MaxRetries,
		retryDelay: opts.RetryDelay,
	}
}

// Retrieve returns the top-k most relevant documents for the query.
func (c *ColBERTv2) Retrieve(ctx context.Context, query string, k int) ([]string, error) {
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
func (c *ColBERTv2) RetrieveWithScores(ctx context.Context, query string, k int) ([]Document, error) {
	// Check cache first
	cacheKey := fmt.Sprintf("%s:%d", query, k)
	c.cacheMutex.RLock()
	if cached, ok := c.cache[cacheKey]; ok {
		c.cacheMutex.RUnlock()
		return cached, nil
	}
	c.cacheMutex.RUnlock()
	
	// Perform retrieval
	var docs []Document
	var err error
	
	for attempt := 0; attempt < c.maxRetries; attempt++ {
		if c.usePost {
			docs, err = c.retrievePost(ctx, query, k)
		} else {
			docs, err = c.retrieveGet(ctx, query, k)
		}
		
		if err == nil {
			break
		}
		
		if attempt < c.maxRetries-1 {
			select {
			case <-time.After(c.retryDelay):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}
	
	if err != nil {
		return nil, err
	}
	
	// Cache the results
	c.cacheMutex.Lock()
	c.cache[cacheKey] = docs
	c.cacheMutex.Unlock()
	
	return docs, nil
}

// retrieveGet performs a GET request to the ColBERTv2 API.
func (c *ColBERTv2) retrieveGet(ctx context.Context, query string, k int) ([]Document, error) {
	if k > 100 {
		return nil, fmt.Errorf("k must be <= 100 for hosted ColBERTv2 server")
	}
	
	params := url.Values{}
	params.Set("query", query)
	params.Set("k", fmt.Sprintf("%d", k))
	
	reqURL := fmt.Sprintf("%s?%s", c.url, params.Encode())
	
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
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
		TopK []map[string]interface{} `json:"topk"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}
	
	return c.parseResults(result.TopK, k)
}

// retrievePost performs a POST request to the ColBERTv2 API.
func (c *ColBERTv2) retrievePost(ctx context.Context, query string, k int) ([]Document, error) {
	payload := map[string]interface{}{
		"query": query,
		"k":     k,
	}
	
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %w", err)
	}
	
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json; charset=utf-8")
	
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	
	var result struct {
		TopK []map[string]interface{} `json:"topk"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}
	
	return c.parseResults(result.TopK, k)
}

// parseResults converts the API response to Document objects.
func (c *ColBERTv2) parseResults(topk []map[string]interface{}, k int) ([]Document, error) {
	docs := make([]Document, 0, len(topk))
	
	for i, item := range topk {
		if i >= k {
			break
		}
		
		doc := Document{
			Metadata: item,
		}
		
		// Extract text (could be "text" or "long_text")
		if text, ok := item["text"].(string); ok {
			doc.Content = text
		} else if longText, ok := item["long_text"].(string); ok {
			doc.Content = longText
		}
		
		// Extract score
		if score, ok := item["score"].(float64); ok {
			doc.Score = score
		}
		
		// Extract ID
		if id, ok := item["pid"].(float64); ok {
			doc.ID = fmt.Sprintf("%d", int(id))
		} else if id, ok := item["id"].(string); ok {
			doc.ID = id
		}
		
		docs = append(docs, doc)
	}
	
	return docs, nil
}

// BatchRetrieve retrieves documents for multiple queries in parallel.
func (c *ColBERTv2) BatchRetrieve(ctx context.Context, queries []string, k int) ([][]Document, error) {
	results := make([][]Document, len(queries))
	errors := make([]error, len(queries))
	
	var wg sync.WaitGroup
	for i, query := range queries {
		wg.Add(1)
		go func(idx int, q string) {
			defer wg.Done()
			docs, err := c.RetrieveWithScores(ctx, q, k)
			results[idx] = docs
			errors[idx] = err
		}(i, query)
	}
	
	wg.Wait()
	
	// Check for errors
	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}
	
	return results, nil
}

// ClearCache clears the retrieval cache.
func (c *ColBERTv2) ClearCache() {
	c.cacheMutex.Lock()
	defer c.cacheMutex.Unlock()
	c.cache = make(map[string][]Document)
}
