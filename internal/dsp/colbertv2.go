// Package dsp provides legacy DSP compatibility layer.
package dsp

import (
	"context"

	"github.com/stanfordnlp/dspy/internal/retrievers"
)

// ColBERTv2 provides a legacy-compatible interface to ColBERTv2 retriever.
type ColBERTv2 struct {
	retriever *retrievers.ColBERTv2
}

// NewColBERTv2 creates a new ColBERTv2 instance with legacy interface.
func NewColBERTv2(url string, port string, usePost bool) *ColBERTv2 {
	opts := retrievers.DefaultColBERTv2Options()
	opts.URL = url
	opts.Port = port
	opts.UsePost = usePost
	
	return &ColBERTv2{
		retriever: retrievers.NewColBERTv2(opts),
	}
}

// Call retrieves documents for a query (legacy interface).
func (c *ColBERTv2) Call(query string, k int) ([]map[string]interface{}, error) {
	ctx := context.Background()
	docs, err := c.retriever.RetrieveWithScores(ctx, query, k)
	if err != nil {
		return nil, err
	}
	
	// Convert to legacy format
	results := make([]map[string]interface{}, len(docs))
	for i, doc := range docs {
		results[i] = map[string]interface{}{
			"long_text": doc.Content,
			"score":     doc.Score,
			"pid":       doc.ID,
		}
		
		// Merge any additional metadata
		for k, v := range doc.Metadata {
			if _, exists := results[i][k]; !exists {
				results[i][k] = v
			}
		}
	}
	
	return results, nil
}

// __call__ provides Python-like callable interface.
func (c *ColBERTv2) __call__(query string, k int, simplify bool) (interface{}, error) {
	docs, err := c.Call(query, k)
	if err != nil {
		return nil, err
	}
	
	if simplify {
		// Return just the text
		texts := make([]string, len(docs))
		for i, doc := range docs {
			if text, ok := doc["long_text"].(string); ok {
				texts[i] = text
			}
		}
		return texts, nil
	}
	
	return docs, nil
}

// ClearCache clears the retriever cache.
func (c *ColBERTv2) ClearCache() {
	c.retriever.ClearCache()
}
