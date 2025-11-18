// Package cache provides caching functionality for LM responses.
package cache

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// Cache defines the interface for response caching.
type Cache interface {
	// Get retrieves a cached response.
	Get(ctx context.Context, key string) ([]byte, bool, error)

	// Set stores a response in the cache.
	Set(ctx context.Context, key string, value []byte, ttl time.Duration) error

	// Delete removes a response from the cache.
	Delete(ctx context.Context, key string) error

	// Clear removes all cached responses.
	Clear(ctx context.Context) error
}

// MemoryCache is a simple in-memory cache implementation.
type MemoryCache struct {
	store map[string]*cacheEntry
	mu    sync.RWMutex
}

type cacheEntry struct {
	value     []byte
	expiresAt time.Time
}

// NewMemoryCache creates a new in-memory cache.
func NewMemoryCache() *MemoryCache {
	cache := &MemoryCache{
		store: make(map[string]*cacheEntry),
	}

	// Start cleanup goroutine
	go cache.cleanup()

	return cache
}

// Get implements Cache.Get.
func (c *MemoryCache) Get(ctx context.Context, key string) ([]byte, bool, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, ok := c.store[key]
	if !ok {
		return nil, false, nil
	}

	// Check if expired
	if time.Now().After(entry.expiresAt) {
		return nil, false, nil
	}

	return entry.value, true, nil
}

// Set implements Cache.Set.
func (c *MemoryCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	expiresAt := time.Now().Add(ttl)
	if ttl == 0 {
		expiresAt = time.Now().Add(24 * time.Hour) // Default 24h
	}

	c.store[key] = &cacheEntry{
		value:     value,
		expiresAt: expiresAt,
	}

	return nil
}

// Delete implements Cache.Delete.
func (c *MemoryCache) Delete(ctx context.Context, key string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.store, key)
	return nil
}

// Clear implements Cache.Clear.
func (c *MemoryCache) Clear(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.store = make(map[string]*cacheEntry)
	return nil
}

// cleanup periodically removes expired entries.
func (c *MemoryCache) cleanup() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		c.mu.Lock()
		now := time.Now()
		for key, entry := range c.store {
			if now.After(entry.expiresAt) {
				delete(c.store, key)
			}
		}
		c.mu.Unlock()
	}
}

// GenerateCacheKey generates a cache key from a request.
func GenerateCacheKey(data interface{}) (string, error) {
	// Serialize the data
	jsonData, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("failed to serialize for cache key: %w", err)
	}

	// Hash the serialized data
	hash := sha256.Sum256(jsonData)
	return fmt.Sprintf("%x", hash), nil
}
