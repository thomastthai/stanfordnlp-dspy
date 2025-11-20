package dspy

import (
	"sync"
	"time"
)

// Settings holds the global configuration for DSPy.
type Settings struct {
	// LM is the default language model to use
	LM interface{}

	// RM is the default retrieval model to use
	RM interface{}

	// Adapter is the format adapter to use
	Adapter string

	// Temperature for sampling
	Temperature float64

	// MaxTokens is the maximum number of tokens to generate
	MaxTokens int

	// CacheDir is the directory for disk caching
	// Deprecated: Use CacheConfig.Dir instead
	CacheDir string

	// EnableCache enables/disables caching
	// Deprecated: Use CacheConfig.Enabled instead
	EnableCache bool

	// CacheConfig controls caching behavior
	CacheConfig *CacheConfig

	// Timeout for LM requests
	Timeout time.Duration

	// MaxRetries for failed requests
	MaxRetries int

	// Trace enables detailed execution tracing
	Trace bool

	// Experimental features
	Experimental map[string]interface{}

	mu sync.RWMutex
}

// NewSettings creates a new Settings instance with default values.
func NewSettings() *Settings {
	cacheConfig := DefaultCacheConfig()
	return &Settings{
		Temperature:  0.0,
		MaxTokens:    1000,
		CacheDir:     cacheConfig.Dir,     // Backward compatibility
		EnableCache:  cacheConfig.Enabled, // Backward compatibility
		CacheConfig:  cacheConfig,
		Timeout:      30 * time.Second,
		MaxRetries:   3,
		Trace:        false,
		Experimental: make(map[string]interface{}),
	}
}

// Copy creates a deep copy of the settings.
func (s *Settings) Copy() *Settings {
	s.mu.RLock()
	defer s.mu.RUnlock()

	exp := make(map[string]interface{}, len(s.Experimental))
	for k, v := range s.Experimental {
		exp[k] = v
	}

	// Deep copy CacheConfig
	var cacheConfig *CacheConfig
	if s.CacheConfig != nil {
		cacheConfig = &CacheConfig{
			Dir:       s.CacheConfig.Dir,
			MaxSizeMB: s.CacheConfig.MaxSizeMB,
			TTL:       s.CacheConfig.TTL,
			Enabled:   s.CacheConfig.Enabled,
		}
	}

	return &Settings{
		LM:           s.LM,
		RM:           s.RM,
		Adapter:      s.Adapter,
		Temperature:  s.Temperature,
		MaxTokens:    s.MaxTokens,
		CacheDir:     s.CacheDir,
		EnableCache:  s.EnableCache,
		CacheConfig:  cacheConfig,
		Timeout:      s.Timeout,
		MaxRetries:   s.MaxRetries,
		Trace:        s.Trace,
		Experimental: exp,
	}
}

// SettingsOption is a functional option for configuring Settings.
type SettingsOption func(*Settings)

// WithLM sets the default language model.
func WithLM(lm interface{}) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.LM = lm
	}
}

// WithRM sets the default retrieval model.
func WithRM(rm interface{}) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.RM = rm
	}
}

// WithAdapter sets the format adapter.
func WithAdapter(adapter string) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.Adapter = adapter
	}
}

// WithTemperature sets the sampling temperature.
func WithTemperature(temp float64) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.Temperature = temp
	}
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(max int) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.MaxTokens = max
	}
}

// WithCacheDir sets the cache directory.
// Deprecated: This function is maintained for backward compatibility.
// The preferred way is to use the WithCacheDir function from cache.go.
func WithCacheDir(dir string) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.CacheDir = dir
		if s.CacheConfig != nil {
			s.CacheConfig.Dir = dir
		}
	}
}

// WithCache enables or disables caching.
func WithCache(enabled bool) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.EnableCache = enabled
		if s.CacheConfig != nil {
			s.CacheConfig.Enabled = enabled
		}
	}
}

// WithTimeout sets the request timeout.
func WithTimeout(timeout time.Duration) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.Timeout = timeout
	}
}

// WithMaxRetries sets the maximum number of retries.
func WithMaxRetries(max int) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.MaxRetries = max
	}
}

// WithTrace enables or disables tracing.
func WithTrace(enabled bool) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.Trace = enabled
	}
}

// WithExperimental sets an experimental feature flag.
func WithExperimental(key string, value interface{}) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.Experimental[key] = value
	}
}
