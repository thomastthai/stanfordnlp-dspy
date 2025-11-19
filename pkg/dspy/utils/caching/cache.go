// Package caching provides caching utilities for DSPy.
package caching

import (
	"context"
	"sync"
	"time"
)

// Cache is a generic interface for caching.
type Cache interface {
	// Get retrieves a value from the cache
	Get(key string) (interface{}, bool)
	
	// Set stores a value in the cache
	Set(key string, value interface{}, ttl time.Duration)
	
	// Delete removes a value from the cache
	Delete(key string)
	
	// Clear removes all values from the cache
	Clear()
	
	// Len returns the number of cached items
	Len() int
}

// cacheEntry holds a cached value with expiration.
type cacheEntry struct {
	value      interface{}
	expiration time.Time
}

// SimpleCache is a simple in-memory cache.
type SimpleCache struct {
	mu    sync.RWMutex
	items map[string]cacheEntry
}

// NewSimpleCache creates a new simple cache.
func NewSimpleCache() *SimpleCache {
	c := &SimpleCache{
		items: make(map[string]cacheEntry),
	}
	
	// Start cleanup goroutine
	go c.cleanupLoop(context.Background())
	
	return c
}

// Get implements Cache.Get.
func (c *SimpleCache) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	entry, ok := c.items[key]
	if !ok {
		return nil, false
	}
	
	// Check expiration
	if !entry.expiration.IsZero() && time.Now().After(entry.expiration) {
		return nil, false
	}
	
	return entry.value, true
}

// Set implements Cache.Set.
func (c *SimpleCache) Set(key string, value interface{}, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	entry := cacheEntry{
		value: value,
	}
	
	if ttl > 0 {
		entry.expiration = time.Now().Add(ttl)
	}
	
	c.items[key] = entry
}

// Delete implements Cache.Delete.
func (c *SimpleCache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	delete(c.items, key)
}

// Clear implements Cache.Clear.
func (c *SimpleCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	c.items = make(map[string]cacheEntry)
}

// Len implements Cache.Len.
func (c *SimpleCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	return len(c.items)
}

// cleanupLoop periodically removes expired entries.
func (c *SimpleCache) cleanupLoop(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.cleanup()
		}
	}
}

// cleanup removes expired entries.
func (c *SimpleCache) cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	now := time.Now()
	for key, entry := range c.items {
		if !entry.expiration.IsZero() && now.After(entry.expiration) {
			delete(c.items, key)
		}
	}
}
