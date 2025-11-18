// Package retrievers provides retrieval implementations for DSPy.
package retrievers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// ColBERTv2 is a retriever that uses the ColBERTv2 API for dense passage retrieval.
type ColBERTv2 struct {
	*BaseRetriever
	url         string
	client      *http.Client
	usePost     bool
	maxK        int
}

// ColBERTv2Options configures a ColBERTv2 retriever.
type ColBERTv2Options struct {
	// URL is the base URL of the ColBERTv2 server
	URL string
	// Port is the port of the ColBERTv2 server (optional)
	Port int
	// UsePost determines whether to use POST or GET requests
	UsePost bool
	// Timeout is the HTTP request timeout
	Timeout time.Duration
	// MaxK is the maximum number of results supported by the server
	MaxK int
}

// colbertResponse represents the API response from ColBERTv2.
type colbertResponse struct {
	TopK []colbertDocument `json:"topk"`
}

// colbertDocument represents a document returned by ColBERTv2.
type colbertDocument struct {
	Text     string                 `json:"text"`
	LongText string                 `json:"long_text"`
	PID      interface{}            `json:"pid"`
	Score    float64                `json:"score"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// NewColBERTv2 creates a new ColBERTv2 retriever.
func NewColBERTv2(opts ColBERTv2Options) *ColBERTv2 {
	if opts.URL == "" {
		opts.URL = "http://0.0.0.0"
	}
	if opts.Port > 0 {
		opts.URL = fmt.Sprintf("%s:%d", opts.URL, opts.Port)
	}
	if opts.Timeout == 0 {
		opts.Timeout = 10 * time.Second
	}
	if opts.MaxK == 0 {
		opts.MaxK = 100
	}

	return &ColBERTv2{
		BaseRetriever: NewBaseRetriever("colbertv2"),
		url:           opts.URL,
		usePost:       opts.UsePost,
		maxK:          opts.MaxK,
		client: &http.Client{
			Timeout: opts.Timeout,
		},
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

// RetrieveWithScores returns documents with their relevance scores.
func (c *ColBERTv2) RetrieveWithScores(ctx context.Context, query string, k int) ([]Document, error) {
	if k > c.maxK {
		return nil, fmt.Errorf("k=%d exceeds maximum supported value of %d", k, c.maxK)
	}

	var resp *colbertResponse
	var err error

	if c.usePost {
		resp, err = c.postRequest(ctx, query, k)
	} else {
		resp, err = c.getRequest(ctx, query, k)
	}

	if err != nil {
		return nil, err
	}

	// Convert to Document format
	docs := make([]Document, 0, len(resp.TopK))
	for _, d := range resp.TopK {
		content := d.LongText
		if content == "" {
			content = d.Text
		}

		// Convert PID to string
		var id string
		if d.PID != nil {
			id = fmt.Sprintf("%v", d.PID)
		}

		docs = append(docs, Document{
			Content:  content,
			Score:    d.Score,
			ID:       id,
			Metadata: d.Metadata,
		})
	}

	return docs, nil
}

// getRequest performs a GET request to the ColBERTv2 API.
func (c *ColBERTv2) getRequest(ctx context.Context, query string, k int) (*colbertResponse, error) {
	params := url.Values{}
	params.Add("query", query)
	params.Add("k", fmt.Sprintf("%d", k))

	reqURL := fmt.Sprintf("%s?%s", c.url, params.Encode())

	req, err := http.NewRequestWithContext(ctx, "GET", reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var result colbertResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	return &result, nil
}

// postRequest performs a POST request to the ColBERTv2 API.
func (c *ColBERTv2) postRequest(ctx context.Context, query string, k int) (*colbertResponse, error) {
	payload := map[string]interface{}{
		"query": query,
		"k":     k,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json; charset=utf-8")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var result colbertResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	return &result, nil
}
