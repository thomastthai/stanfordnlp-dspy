// Package embeddings provides text embedding generation from various providers.
package embeddings

import (
	"context"
	"fmt"
	"sync"
)

// Embedder is the interface for generating text embeddings.
type Embedder interface {
	// Embed generates embeddings for the given texts
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	
	// Dimension returns the dimensionality of the embeddings
	Dimension() int
	
	// MaxBatchSize returns the maximum batch size supported
	MaxBatchSize() int
}

// EmbeddingCache provides caching for embeddings.
type EmbeddingCache interface {
	// Get retrieves an embedding from the cache
	Get(text string) ([]float32, bool)
	
	// Set stores an embedding in the cache
	Set(text string, embedding []float32)
	
	// Clear removes all entries from the cache
	Clear()
	
	// Size returns the number of cached embeddings
	Size() int
}

// SimpleCache is a simple in-memory cache for embeddings.
type SimpleCache struct {
	mu    sync.RWMutex
	cache map[string][]float32
}

// NewSimpleCache creates a new simple embedding cache.
func NewSimpleCache() *SimpleCache {
	return &SimpleCache{
		cache: make(map[string][]float32),
	}
}

// Get retrieves an embedding from the cache.
func (c *SimpleCache) Get(text string) ([]float32, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	emb, ok := c.cache[text]
	return emb, ok
}

// Set stores an embedding in the cache.
func (c *SimpleCache) Set(text string, embedding []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache[text] = embedding
}

// Clear removes all entries from the cache.
func (c *SimpleCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache = make(map[string][]float32)
}

// Size returns the number of cached embeddings.
func (c *SimpleCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.cache)
}

// CachedEmbedder wraps an Embedder with caching.
type CachedEmbedder struct {
	embedder Embedder
	cache    EmbeddingCache
}

// NewCachedEmbedder creates a new cached embedder.
func NewCachedEmbedder(embedder Embedder, cache EmbeddingCache) *CachedEmbedder {
	if cache == nil {
		cache = NewSimpleCache()
	}
	return &CachedEmbedder{
		embedder: embedder,
		cache:    cache,
	}
}

// Embed implements Embedder.Embed with caching.
func (c *CachedEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	uncachedIndices := make([]int, 0)
	uncachedTexts := make([]string, 0)
	
	// Check cache
	for i, text := range texts {
		if emb, ok := c.cache.Get(text); ok {
			results[i] = emb
		} else {
			uncachedIndices = append(uncachedIndices, i)
			uncachedTexts = append(uncachedTexts, text)
		}
	}
	
	// Generate embeddings for uncached texts
	if len(uncachedTexts) > 0 {
		embeddings, err := c.embedder.Embed(ctx, uncachedTexts)
		if err != nil {
			return nil, fmt.Errorf("failed to generate embeddings: %w", err)
		}
		
		// Store in cache and results
		for i, idx := range uncachedIndices {
			c.cache.Set(uncachedTexts[i], embeddings[i])
			results[idx] = embeddings[i]
		}
	}
	
	return results, nil
}

// Dimension implements Embedder.Dimension.
func (c *CachedEmbedder) Dimension() int {
	return c.embedder.Dimension()
}

// MaxBatchSize implements Embedder.MaxBatchSize.
func (c *CachedEmbedder) MaxBatchSize() int {
	return c.embedder.MaxBatchSize()
}

// BatchEmbedder processes texts in batches to respect API limits.
type BatchEmbedder struct {
	embedder  Embedder
	batchSize int
}

// NewBatchEmbedder creates a new batch embedder.
func NewBatchEmbedder(embedder Embedder, batchSize int) *BatchEmbedder {
	if batchSize <= 0 {
		batchSize = embedder.MaxBatchSize()
	}
	return &BatchEmbedder{
		embedder:  embedder,
		batchSize: batchSize,
	}
}

// Embed implements Embedder.Embed with batching.
func (b *BatchEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	
	results := make([][]float32, 0, len(texts))
	
	// Process in batches
	for i := 0; i < len(texts); i += b.batchSize {
		end := i + b.batchSize
		if end > len(texts) {
			end = len(texts)
		}
		
		batch := texts[i:end]
		embeddings, err := b.embedder.Embed(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("failed to embed batch: %w", err)
		}
		
		results = append(results, embeddings...)
	}
	
	return results, nil
}

// Dimension implements Embedder.Dimension.
func (b *BatchEmbedder) Dimension() int {
	return b.embedder.Dimension()
}

// MaxBatchSize implements Embedder.MaxBatchSize.
func (b *BatchEmbedder) MaxBatchSize() int {
	return b.batchSize
}
