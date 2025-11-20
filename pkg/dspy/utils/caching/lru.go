package caching

import (
	"sync"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
)

// LRUCache is an LRU (Least Recently Used) cache.
type LRUCache struct {
	mu    sync.RWMutex
	cache *lru.Cache[string, interface{}]
}

// NewLRUCache creates a new LRU cache with the specified size.
func NewLRUCache(size int) (*LRUCache, error) {
	cache, err := lru.New[string, interface{}](size)
	if err != nil {
		return nil, err
	}

	return &LRUCache{
		cache: cache,
	}, nil
}

// Get implements Cache.Get.
func (c *LRUCache) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.cache.Get(key)
}

// Set implements Cache.Set.
// Note: LRU cache does not support TTL, it evicts based on usage.
func (c *LRUCache) Set(key string, value interface{}, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache.Add(key, value)
}

// Delete implements Cache.Delete.
func (c *LRUCache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache.Remove(key)
}

// Clear implements Cache.Clear.
func (c *LRUCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache.Purge()
}

// Len implements Cache.Len.
func (c *LRUCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.cache.Len()
}
