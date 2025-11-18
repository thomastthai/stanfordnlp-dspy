package cache

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDiskCache_BasicOperations(t *testing.T) {
	// Create temporary cache directory
	tmpDir := filepath.Join(os.TempDir(), "dspy-cache-test")
	defer os.RemoveAll(tmpDir)

	cache, err := NewDiskCache(DiskCacheOptions{
		CachePath: tmpDir,
		MaxSize:   0, // Unlimited
	})
	if err != nil {
		t.Fatalf("Failed to create disk cache: %v", err)
	}
	defer cache.Close()

	ctx := context.Background()

	// Test Set and Get
	key := "test-key"
	value := []byte("test-value")

	err = cache.Set(ctx, key, value, 1*time.Hour)
	if err != nil {
		t.Fatalf("Failed to set cache entry: %v", err)
	}

	retrieved, found, err := cache.Get(ctx, key)
	if err != nil {
		t.Fatalf("Failed to get cache entry: %v", err)
	}
	if !found {
		t.Fatal("Cache entry not found")
	}
	if string(retrieved) != string(value) {
		t.Fatalf("Expected %s, got %s", value, retrieved)
	}

	// Test Delete
	err = cache.Delete(ctx, key)
	if err != nil {
		t.Fatalf("Failed to delete cache entry: %v", err)
	}

	_, found, err = cache.Get(ctx, key)
	if err != nil {
		t.Fatalf("Failed to get deleted entry: %v", err)
	}
	if found {
		t.Fatal("Deleted entry still found")
	}
}

func TestDiskCache_TTL(t *testing.T) {
	tmpDir := filepath.Join(os.TempDir(), "dspy-cache-ttl-test")
	defer os.RemoveAll(tmpDir)

	cache, err := NewDiskCache(DiskCacheOptions{
		CachePath: tmpDir,
	})
	if err != nil {
		t.Fatalf("Failed to create disk cache: %v", err)
	}
	defer cache.Close()

	ctx := context.Background()
	key := "ttl-key"
	value := []byte("ttl-value")

	// Set with short TTL (use 2 seconds for more reliable testing)
	err = cache.Set(ctx, key, value, 2*time.Second)
	if err != nil {
		t.Fatalf("Failed to set cache entry: %v", err)
	}

	// Should be available immediately
	retrieved, found, err := cache.Get(ctx, key)
	if err != nil {
		t.Fatalf("Failed to get cache entry: %v", err)
	}
	if !found {
		t.Fatal("Cache entry not found")
	}
	if string(retrieved) != string(value) {
		t.Fatalf("Expected %s, got %s", value, retrieved)
	}

	// Wait for TTL to expire
	time.Sleep(3 * time.Second)

	// Should be expired now - Badger handles TTL automatically during Get
	_, found, err = cache.Get(ctx, key)
	if err != nil {
		t.Fatalf("Failed to get expired entry: %v", err)
	}
	if found {
		t.Fatal("Expired entry still found")
	}
}

func TestDiskCache_Clear(t *testing.T) {
	tmpDir := filepath.Join(os.TempDir(), "dspy-cache-clear-test")
	defer os.RemoveAll(tmpDir)

	cache, err := NewDiskCache(DiskCacheOptions{
		CachePath: tmpDir,
	})
	if err != nil {
		t.Fatalf("Failed to create disk cache: %v", err)
	}
	defer cache.Close()

	ctx := context.Background()

	// Add multiple entries
	for i := 0; i < 5; i++ {
		key := string(rune('a' + i))
		value := []byte("value-" + key)
		err = cache.Set(ctx, key, value, 1*time.Hour)
		if err != nil {
			t.Fatalf("Failed to set entry %s: %v", key, err)
		}
	}

	// Clear all
	err = cache.Clear(ctx)
	if err != nil {
		t.Fatalf("Failed to clear cache: %v", err)
	}

	// Verify all entries are gone
	for i := 0; i < 5; i++ {
		key := string(rune('a' + i))
		_, found, err := cache.Get(ctx, key)
		if err != nil {
			t.Fatalf("Failed to check entry %s: %v", key, err)
		}
		if found {
			t.Fatalf("Entry %s still found after clear", key)
		}
	}
}

func TestDiskCache_Stats(t *testing.T) {
	tmpDir := filepath.Join(os.TempDir(), "dspy-cache-stats-test")
	defer os.RemoveAll(tmpDir)

	cache, err := NewDiskCache(DiskCacheOptions{
		CachePath: tmpDir,
	})
	if err != nil {
		t.Fatalf("Failed to create disk cache: %v", err)
	}
	defer cache.Close()

	ctx := context.Background()
	key := "stats-key"
	value := []byte("stats-value")

	// Set entry
	err = cache.Set(ctx, key, value, 1*time.Hour)
	if err != nil {
		t.Fatalf("Failed to set entry: %v", err)
	}

	// Get initial stats (after set)
	initialStats := cache.Stats()

	// Hit
	_, found, err := cache.Get(ctx, key)
	if err != nil {
		t.Fatalf("Failed to get entry: %v", err)
	}
	if !found {
		t.Fatal("Entry should be found")
	}

	// Miss
	_, found, err = cache.Get(ctx, "nonexistent-key")
	if err != nil {
		t.Fatalf("Failed to get nonexistent entry: %v", err)
	}
	if found {
		t.Fatal("Entry should not be found")
	}

	stats := cache.Stats()
	if stats.Hits != initialStats.Hits+1 {
		t.Fatalf("Expected %d hits, got %d", initialStats.Hits+1, stats.Hits)
	}
	if stats.Misses != initialStats.Misses+1 {
		t.Fatalf("Expected %d misses, got %d", initialStats.Misses+1, stats.Misses)
	}
}

func TestDiskCache_Persistence(t *testing.T) {
	tmpDir := filepath.Join(os.TempDir(), "dspy-cache-persist-test")
	defer os.RemoveAll(tmpDir)

	ctx := context.Background()
	key := "persist-key"
	value := []byte("persist-value")

	// Create cache and set a value
	cache1, err := NewDiskCache(DiskCacheOptions{
		CachePath: tmpDir,
	})
	if err != nil {
		t.Fatalf("Failed to create disk cache: %v", err)
	}

	err = cache1.Set(ctx, key, value, 1*time.Hour)
	if err != nil {
		t.Fatalf("Failed to set cache entry: %v", err)
	}
	cache1.Close()

	// Open a new cache instance with the same path
	cache2, err := NewDiskCache(DiskCacheOptions{
		CachePath: tmpDir,
	})
	if err != nil {
		t.Fatalf("Failed to create second disk cache: %v", err)
	}
	defer cache2.Close()

	// Value should still be there
	retrieved, found, err := cache2.Get(ctx, key)
	if err != nil {
		t.Fatalf("Failed to get cache entry: %v", err)
	}
	if !found {
		t.Fatal("Cache entry not found after restart")
	}
	if string(retrieved) != string(value) {
		t.Fatalf("Expected %s, got %s", value, retrieved)
	}
}
