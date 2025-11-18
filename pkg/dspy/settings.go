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
	CacheDir string

	// EnableCache enables/disables caching
	EnableCache bool

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
	return &Settings{
		Temperature:  0.0,
		MaxTokens:    1000,
		CacheDir:     ".dspy_cache",
		EnableCache:  true,
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

	return &Settings{
		LM:           s.LM,
		RM:           s.RM,
		Adapter:      s.Adapter,
		Temperature:  s.Temperature,
		MaxTokens:    s.MaxTokens,
		CacheDir:     s.CacheDir,
		EnableCache:  s.EnableCache,
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
func WithCacheDir(dir string) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.CacheDir = dir
	}
}

// WithCache enables or disables caching.
func WithCache(enabled bool) SettingsOption {
	return func(s *Settings) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.EnableCache = enabled
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
