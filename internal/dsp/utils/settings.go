// Package utils provides DSP utility functions.
package utils

import (
	"context"
	"sync"
)

// Settings holds global DSP settings.
type Settings struct {
	LM              interface{}
	RM              interface{}
	Adapter         interface{}
	Trace           bool
	TraceDir        string
	CacheDir        string
	AsyncMaxWorkers int
	mu              sync.RWMutex
}

var (
	globalSettings *Settings
	settingsMutex  sync.RWMutex
)

func init() {
	globalSettings = NewSettings()
}

// NewSettings creates a new Settings instance.
func NewSettings() *Settings {
	return &Settings{
		AsyncMaxWorkers: 10,
		Trace:           false,
		TraceDir:        "./traces",
		CacheDir:        "./.cache",
	}
}

// GetSettings returns the global settings.
func GetSettings() *Settings {
	settingsMutex.RLock()
	defer settingsMutex.RUnlock()
	return globalSettings
}

// SetSettings sets the global settings.
func SetSettings(settings *Settings) {
	settingsMutex.Lock()
	defer settingsMutex.Unlock()
	globalSettings = settings
}

// SetLM sets the language model.
func (s *Settings) SetLM(lm interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.LM = lm
}

// GetLM gets the language model.
func (s *Settings) GetLM() interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.LM
}

// SetRM sets the retrieval model.
func (s *Settings) SetRM(rm interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.RM = rm
}

// GetRM gets the retrieval model.
func (s *Settings) GetRM() interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.RM
}

// SetAdapter sets the adapter.
func (s *Settings) SetAdapter(adapter interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Adapter = adapter
}

// GetAdapter gets the adapter.
func (s *Settings) GetAdapter() interface{} {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.Adapter
}

// SetTrace enables or disables tracing.
func (s *Settings) SetTrace(trace bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Trace = trace
}

// IsTraceEnabled checks if tracing is enabled.
func (s *Settings) IsTraceEnabled() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.Trace
}

// SetTraceDir sets the trace directory.
func (s *Settings) SetTraceDir(dir string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.TraceDir = dir
}

// GetTraceDir gets the trace directory.
func (s *Settings) GetTraceDir() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.TraceDir
}

// SetCacheDir sets the cache directory.
func (s *Settings) SetCacheDir(dir string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.CacheDir = dir
}

// GetCacheDir gets the cache directory.
func (s *Settings) GetCacheDir() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.CacheDir
}

// SetAsyncMaxWorkers sets the maximum number of async workers.
func (s *Settings) SetAsyncMaxWorkers(workers int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.AsyncMaxWorkers = workers
}

// GetAsyncMaxWorkers gets the maximum number of async workers.
func (s *Settings) GetAsyncMaxWorkers() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.AsyncMaxWorkers
}

// Copy creates a copy of the settings.
func (s *Settings) Copy() *Settings {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	return &Settings{
		LM:              s.LM,
		RM:              s.RM,
		Adapter:         s.Adapter,
		Trace:           s.Trace,
		TraceDir:        s.TraceDir,
		CacheDir:        s.CacheDir,
		AsyncMaxWorkers: s.AsyncMaxWorkers,
	}
}

// Configure configures global DSP settings.
func Configure(opts ...SettingOption) {
	settings := GetSettings()
	for _, opt := range opts {
		opt(settings)
	}
}

// SettingOption is a function that modifies settings.
type SettingOption func(*Settings)

// WithLM sets the language model.
func WithLM(lm interface{}) SettingOption {
	return func(s *Settings) {
		s.SetLM(lm)
	}
}

// WithRM sets the retrieval model.
func WithRM(rm interface{}) SettingOption {
	return func(s *Settings) {
		s.SetRM(rm)
	}
}

// WithTrace enables tracing.
func WithTrace(enabled bool) SettingOption {
	return func(s *Settings) {
		s.SetTrace(enabled)
	}
}

// WithAsyncMaxWorkers sets the max async workers.
func WithAsyncMaxWorkers(workers int) SettingOption {
	return func(s *Settings) {
		s.SetAsyncMaxWorkers(workers)
	}
}

// ContextKey is the type for context keys.
type ContextKey string

const (
	// SettingsContextKey is the context key for settings.
	SettingsContextKey ContextKey = "dsp_settings"
)

// WithContext adds settings to a context.
func (s *Settings) WithContext(ctx context.Context) context.Context {
	return context.WithValue(ctx, SettingsContextKey, s)
}

// FromContext extracts settings from a context.
func FromContext(ctx context.Context) *Settings {
	if settings, ok := ctx.Value(SettingsContextKey).(*Settings); ok {
		return settings
	}
	return GetSettings()
}
