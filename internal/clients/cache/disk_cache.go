// Package cache provides caching functionality for LM responses.
package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/dgraph-io/badger/v3"
)

// DiskCache is a persistent cache implementation using Badger DB.
type DiskCache struct {
	db        *badger.DB
	mu        sync.RWMutex
	stats     CacheStats
	maxSize   int64 // Maximum cache size in bytes (0 = unlimited)
	cachePath string
}

// CacheStats tracks cache performance metrics.
type CacheStats struct {
	Hits      int64
	Misses    int64
	Evictions int64
	Errors    int64
}

// DiskCacheOptions configures the disk cache.
type DiskCacheOptions struct {
	// CachePath is the directory where cache data is stored
	CachePath string
	// MaxSize is the maximum cache size in bytes (0 = unlimited)
	MaxSize int64
	// TTL is the default time-to-live for cache entries
	TTL time.Duration
}

// NewDiskCache creates a new disk-backed cache using Badger.
func NewDiskCache(opts DiskCacheOptions) (*DiskCache, error) {
	if opts.CachePath == "" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("failed to get home directory: %w", err)
		}
		opts.CachePath = filepath.Join(homeDir, ".dspy", "cache")
	}

	// Create cache directory if it doesn't exist
	if err := os.MkdirAll(opts.CachePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create cache directory: %w", err)
	}

	// Open Badger database
	badgerOpts := badger.DefaultOptions(opts.CachePath)
	badgerOpts.Logger = nil // Disable Badger's default logging

	db, err := badger.Open(badgerOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to open badger database: %w", err)
	}

	cache := &DiskCache{
		db:        db,
		maxSize:   opts.MaxSize,
		cachePath: opts.CachePath,
	}

	// Start garbage collection goroutine
	go cache.runGC()

	return cache, nil
}

// Get retrieves a cached response.
func (c *DiskCache) Get(ctx context.Context, key string) ([]byte, bool, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var value []byte
	err := c.db.View(func(txn *badger.Txn) error {
		item, err := txn.Get([]byte(key))
		if err != nil {
			return err
		}

		value, err = item.ValueCopy(nil)
		return err
	})

	if err != nil {
		if err == badger.ErrKeyNotFound {
			c.stats.Misses++
			return nil, false, nil
		}
		c.stats.Errors++
		return nil, false, fmt.Errorf("failed to get from cache: %w", err)
	}

	c.stats.Hits++
	return value, true, nil
}

// Set stores a response in the cache.
func (c *DiskCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check size limits if configured
	if c.maxSize > 0 {
		if err := c.enforceMaxSize(); err != nil {
			c.stats.Errors++
			return fmt.Errorf("failed to enforce max size: %w", err)
		}
	}

	err := c.db.Update(func(txn *badger.Txn) error {
		entry := badger.NewEntry([]byte(key), value)
		if ttl > 0 {
			entry = entry.WithTTL(ttl)
		} else {
			entry = entry.WithTTL(24 * time.Hour) // Default 24h
		}
		return txn.SetEntry(entry)
	})

	if err != nil {
		c.stats.Errors++
		return fmt.Errorf("failed to set cache entry: %w", err)
	}

	return nil
}

// Delete removes a response from the cache.
func (c *DiskCache) Delete(ctx context.Context, key string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	err := c.db.Update(func(txn *badger.Txn) error {
		return txn.Delete([]byte(key))
	})

	if err != nil {
		c.stats.Errors++
		return fmt.Errorf("failed to delete cache entry: %w", err)
	}

	return nil
}

// Clear removes all cached responses.
func (c *DiskCache) Clear(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	err := c.db.DropAll()
	if err != nil {
		c.stats.Errors++
		return fmt.Errorf("failed to clear cache: %w", err)
	}

	return nil
}

// Close closes the cache and releases resources.
func (c *DiskCache) Close() error {
	return c.db.Close()
}

// Stats returns cache statistics.
func (c *DiskCache) Stats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.stats
}

// enforceMaxSize ensures the cache doesn't exceed maxSize by evicting old entries.
func (c *DiskCache) enforceMaxSize() error {
	if c.maxSize <= 0 {
		return nil
	}

	lsm, vlog := c.db.Size()
	currentSize := lsm + vlog

	if currentSize <= c.maxSize {
		return nil
	}

	// Need to evict entries
	// Collect entries with their timestamps for LRU eviction
	type entry struct {
		key       []byte
		expiresAt uint64
	}
	var entries []entry

	err := c.db.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		opts.PrefetchValues = false
		it := txn.NewIterator(opts)
		defer it.Close()

		for it.Rewind(); it.Valid(); it.Next() {
			item := it.Item()
			entries = append(entries, entry{
				key:       item.KeyCopy(nil),
				expiresAt: item.ExpiresAt(),
			})
		}
		return nil
	})

	if err != nil {
		return err
	}

	// Sort by expiration time (oldest first)
	// Simple bubble sort for now
	for i := 0; i < len(entries)-1; i++ {
		for j := 0; j < len(entries)-i-1; j++ {
			if entries[j].expiresAt > entries[j+1].expiresAt {
				entries[j], entries[j+1] = entries[j+1], entries[j]
			}
		}
	}

	// Evict oldest entries until we're under the limit
	evictCount := len(entries) / 4 // Evict 25% of entries
	if evictCount < 1 {
		evictCount = 1
	}

	err = c.db.Update(func(txn *badger.Txn) error {
		for i := 0; i < evictCount && i < len(entries); i++ {
			if err := txn.Delete(entries[i].key); err != nil {
				return err
			}
			c.stats.Evictions++
		}
		return nil
	})

	return err
}

// runGC periodically runs garbage collection on the database.
func (c *DiskCache) runGC() {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		c.mu.Lock()
		err := c.db.RunValueLogGC(0.5)
		if err != nil && err != badger.ErrNoRewrite {
			// Log error but continue
		}
		c.mu.Unlock()
	}
}

// MarshalCacheEntry marshals data for cache storage.
func MarshalCacheEntry(data interface{}) ([]byte, error) {
	return json.Marshal(data)
}

// UnmarshalCacheEntry unmarshals data from cache.
func UnmarshalCacheEntry(data []byte, v interface{}) error {
	return json.Unmarshal(data, v)
}
