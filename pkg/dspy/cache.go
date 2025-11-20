package dspy

import (
	"os"
	"strconv"
	"time"
)

// CacheConfig controls caching behavior.
type CacheConfig struct {
	// Dir is the cache directory path.
	// Default: $DSPY_CACHE_DIR or os.TempDir()
	Dir string

	// MaxSizeMB is the maximum cache size in megabytes.
	// Default: 1024 (1GB)
	MaxSizeMB int64

	// TTL is how long cached items are valid.
	// Default: 24 hours
	TTL time.Duration

	// Enabled controls whether caching is active.
	// Default: true
	Enabled bool
}

// DefaultCacheConfig returns the default cache configuration.
// It reads from environment variables:
//   - DSPY_CACHE_DIR: Cache directory path (default: os.TempDir())
//   - DSPY_CACHE_SIZE_MB: Max cache size in MB (default: 1024)
//   - DSPY_CACHE_TTL: Cache TTL duration (default: 24h)
//   - DSPY_CACHE_ENABLED: Enable/disable cache (default: true)
func DefaultCacheConfig() *CacheConfig {
	cacheDir := os.Getenv("DSPY_CACHE_DIR")
	if cacheDir == "" {
		cacheDir = os.TempDir()
	}

	sizeMB := int64(1024)
	if sizeStr := os.Getenv("DSPY_CACHE_SIZE_MB"); sizeStr != "" {
		if size, err := strconv.ParseInt(sizeStr, 10, 64); err == nil {
			sizeMB = size
		}
	}

	ttl := 24 * time.Hour
	if ttlStr := os.Getenv("DSPY_CACHE_TTL"); ttlStr != "" {
		if duration, err := time.ParseDuration(ttlStr); err == nil {
			ttl = duration
		}
	}

	enabled := true
	if enabledStr := os.Getenv("DSPY_CACHE_ENABLED"); enabledStr == "false" {
		enabled = false
	}

	return &CacheConfig{
		Dir:       cacheDir,
		MaxSizeMB: sizeMB,
		TTL:       ttl,
		Enabled:   enabled,
	}
}

// WithCacheSize sets the maximum cache size.
func WithCacheSize(sizeMB int64) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.CacheConfig != nil {
			s.CacheConfig.MaxSizeMB = sizeMB
		}
	}
}

// WithCacheTTL sets the cache TTL.
func WithCacheTTL(ttl time.Duration) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.CacheConfig != nil {
			s.CacheConfig.TTL = ttl
		}
	}
}

// DisableCache disables caching.
func DisableCache() SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.CacheConfig != nil {
			s.CacheConfig.Enabled = false
		}
		s.EnableCache = false // Backward compatibility
	}
}
