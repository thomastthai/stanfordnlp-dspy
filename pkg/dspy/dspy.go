// Package dspy provides the public API for DSPy-Go.
//
// DSPy is a framework for programming—rather than prompting—language models.
// It allows you to iterate fast on building modular AI systems and offers
// algorithms for optimizing their prompts and weights.
//
// This Go implementation follows Terraform's architectural patterns and provides:
//   - Interface-based design similar to Terraform providers
//   - Context-based execution for cancellation and timeouts
//   - Goroutines and channels for concurrency
//   - Functional options pattern for configuration
//   - Clean separation of concerns
//
// Example usage:
//
//	import "github.com/stanfordnlp/dspy/pkg/dspy"
//
//	func main() {
//	    // Configure DSPy with an LM
//	    dspy.Configure(
//	        dspy.WithLM("openai/gpt-4"),
//	        dspy.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
//	    )
//
//	    // Use a simple prediction module
//	    predictor := predict.New("question -> answer")
//	    result := predictor.Forward(context.Background(), map[string]interface{}{
//	        "question": "What is DSPy?",
//	    })
//	}
package dspy

import (
	"context"
	"sync"
)

var (
	// Global settings instance
	globalSettings *Settings
	settingsMux    sync.RWMutex
)

func init() {
	globalSettings = NewSettings()
}

// GetSettings returns the global settings instance.
// Thread-safe for concurrent access.
func GetSettings() *Settings {
	settingsMux.RLock()
	defer settingsMux.RUnlock()
	return globalSettings
}

// SetSettings replaces the global settings instance.
// Thread-safe for concurrent access.
func SetSettings(s *Settings) {
	settingsMux.Lock()
	defer settingsMux.Unlock()
	globalSettings = s
}

// Context creates a new context with DSPy settings.
// This allows for request-scoped configuration overrides.
func Context(ctx context.Context, opts ...SettingsOption) context.Context {
	settings := GetSettings().Copy()
	for _, opt := range opts {
		opt(settings)
	}
	return context.WithValue(ctx, settingsKey{}, settings)
}

// contextKey is used to store settings in context
type settingsKey struct{}

// SettingsFromContext extracts settings from context, or returns global settings.
func SettingsFromContext(ctx context.Context) *Settings {
	if s, ok := ctx.Value(settingsKey{}).(*Settings); ok {
		return s
	}
	return GetSettings()
}
