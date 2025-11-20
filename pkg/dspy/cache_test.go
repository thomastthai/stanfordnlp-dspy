package dspy

import (
	"os"
	"testing"
	"time"
)

func TestDefaultCacheConfig(t *testing.T) {
	// Save original env vars
	origDir := os.Getenv("DSPY_CACHE_DIR")
	origSize := os.Getenv("DSPY_CACHE_SIZE_MB")
	origTTL := os.Getenv("DSPY_CACHE_TTL")
	origEnabled := os.Getenv("DSPY_CACHE_ENABLED")
	defer func() {
		os.Setenv("DSPY_CACHE_DIR", origDir)
		os.Setenv("DSPY_CACHE_SIZE_MB", origSize)
		os.Setenv("DSPY_CACHE_TTL", origTTL)
		os.Setenv("DSPY_CACHE_ENABLED", origEnabled)
	}()

	// Test default values (no env vars)
	os.Unsetenv("DSPY_CACHE_DIR")
	os.Unsetenv("DSPY_CACHE_SIZE_MB")
	os.Unsetenv("DSPY_CACHE_TTL")
	os.Unsetenv("DSPY_CACHE_ENABLED")

	config := DefaultCacheConfig()

	if config.Dir == "" {
		t.Error("expected non-empty cache dir")
	}

	if config.MaxSizeMB != 1024 {
		t.Errorf("expected default max size 1024, got %d", config.MaxSizeMB)
	}

	if config.TTL != 24*time.Hour {
		t.Errorf("expected default TTL 24h, got %v", config.TTL)
	}

	if !config.Enabled {
		t.Error("expected cache to be enabled by default")
	}
}

func TestDefaultCacheConfig_WithEnvVars(t *testing.T) {
	// Save original env vars
	origDir := os.Getenv("DSPY_CACHE_DIR")
	origSize := os.Getenv("DSPY_CACHE_SIZE_MB")
	origTTL := os.Getenv("DSPY_CACHE_TTL")
	origEnabled := os.Getenv("DSPY_CACHE_ENABLED")
	defer func() {
		os.Setenv("DSPY_CACHE_DIR", origDir)
		os.Setenv("DSPY_CACHE_SIZE_MB", origSize)
		os.Setenv("DSPY_CACHE_TTL", origTTL)
		os.Setenv("DSPY_CACHE_ENABLED", origEnabled)
	}()

	// Test with custom env vars
	os.Setenv("DSPY_CACHE_DIR", "/custom/cache")
	os.Setenv("DSPY_CACHE_SIZE_MB", "2048")
	os.Setenv("DSPY_CACHE_TTL", "48h")
	os.Setenv("DSPY_CACHE_ENABLED", "false")

	config := DefaultCacheConfig()

	if config.Dir != "/custom/cache" {
		t.Errorf("expected cache dir /custom/cache, got %s", config.Dir)
	}

	if config.MaxSizeMB != 2048 {
		t.Errorf("expected max size 2048, got %d", config.MaxSizeMB)
	}

	if config.TTL != 48*time.Hour {
		t.Errorf("expected TTL 48h, got %v", config.TTL)
	}

	if config.Enabled {
		t.Error("expected cache to be disabled")
	}
}

func TestWithCacheSize(t *testing.T) {
	s := NewSettings()
	opt := WithCacheSize(2048)
	opt(s)

	if s.CacheConfig.MaxSizeMB != 2048 {
		t.Errorf("expected max size 2048, got %d", s.CacheConfig.MaxSizeMB)
	}
}

func TestWithCacheTTL(t *testing.T) {
	s := NewSettings()
	opt := WithCacheTTL(48 * time.Hour)
	opt(s)

	if s.CacheConfig.TTL != 48*time.Hour {
		t.Errorf("expected TTL 48h, got %v", s.CacheConfig.TTL)
	}
}

func TestDisableCache(t *testing.T) {
	s := NewSettings()
	opt := DisableCache()
	opt(s)

	if s.CacheConfig.Enabled {
		t.Error("expected cache to be disabled")
	}

	// Check backward compatibility
	if s.EnableCache {
		t.Error("expected EnableCache to be false for backward compatibility")
	}
}

func TestWithCacheDir_BackwardCompatibility(t *testing.T) {
	s := NewSettings()
	opt := WithCacheDir("/test/cache")
	opt(s)

	// Check both old and new fields are updated
	if s.CacheDir != "/test/cache" {
		t.Errorf("expected CacheDir /test/cache, got %s", s.CacheDir)
	}

	if s.CacheConfig.Dir != "/test/cache" {
		t.Errorf("expected CacheConfig.Dir /test/cache, got %s", s.CacheConfig.Dir)
	}
}

func TestWithCache_BackwardCompatibility(t *testing.T) {
	s := NewSettings()
	opt := WithCache(false)
	opt(s)

	// Check both old and new fields are updated
	if s.EnableCache {
		t.Error("expected EnableCache to be false")
	}

	if s.CacheConfig.Enabled {
		t.Error("expected CacheConfig.Enabled to be false")
	}
}

func TestNewSettings_CacheConfigInitialized(t *testing.T) {
	s := NewSettings()

	if s.CacheConfig == nil {
		t.Fatal("expected CacheConfig to be initialized")
	}

	// Check backward compatibility - deprecated fields should match CacheConfig
	if s.CacheDir != s.CacheConfig.Dir {
		t.Error("CacheDir should match CacheConfig.Dir")
	}

	if s.EnableCache != s.CacheConfig.Enabled {
		t.Error("EnableCache should match CacheConfig.Enabled")
	}
}

func TestSettingsCopy_CacheConfig(t *testing.T) {
	s := NewSettings()
	s.CacheConfig.Dir = "/custom/cache"
	s.CacheConfig.MaxSizeMB = 2048
	s.CacheConfig.TTL = 48 * time.Hour
	s.CacheConfig.Enabled = false

	s2 := s.Copy()

	if s2.CacheConfig.Dir != "/custom/cache" {
		t.Error("CacheConfig.Dir not copied correctly")
	}

	if s2.CacheConfig.MaxSizeMB != 2048 {
		t.Error("CacheConfig.MaxSizeMB not copied correctly")
	}

	if s2.CacheConfig.TTL != 48*time.Hour {
		t.Error("CacheConfig.TTL not copied correctly")
	}

	if s2.CacheConfig.Enabled {
		t.Error("CacheConfig.Enabled not copied correctly")
	}

	// Verify it's a deep copy - modifying copy should not affect original
	s2.CacheConfig.Dir = "/other/cache"
	if s.CacheConfig.Dir == "/other/cache" {
		t.Error("modifying copy affected original")
	}
}
