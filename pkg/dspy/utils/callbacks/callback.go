// Package callbacks provides callback hooks for DSPy operations.
package callbacks

import (
	"context"
	"time"
)

// LMCallbackData contains data passed to LM callbacks.
type LMCallbackData struct {
	Model      string
	Prompt     string
	Response   string
	Tokens     int
	Duration   time.Duration
	Error      error
	Metadata   map[string]interface{}
}

// LMCallback is called before or after an LM call.
type LMCallback func(ctx context.Context, data LMCallbackData)

// ModuleCallbackData contains data passed to module callbacks.
type ModuleCallbackData struct {
	ModuleName string
	Inputs     map[string]interface{}
	Outputs    map[string]interface{}
	Duration   time.Duration
	Error      error
}

// ModuleCallback is called before or after a module execution.
type ModuleCallback func(ctx context.Context, data ModuleCallbackData)

// CallbackManager manages callbacks for DSPy operations.
type CallbackManager struct {
	preLMCallbacks    []LMCallback
	postLMCallbacks   []LMCallback
	preModuleCallbacks  []ModuleCallback
	postModuleCallbacks []ModuleCallback
}

// NewCallbackManager creates a new callback manager.
func NewCallbackManager() *CallbackManager {
	return &CallbackManager{
		preLMCallbacks:      make([]LMCallback, 0),
		postLMCallbacks:     make([]LMCallback, 0),
		preModuleCallbacks:  make([]ModuleCallback, 0),
		postModuleCallbacks: make([]ModuleCallback, 0),
	}
}

// AddPreLMCallback adds a callback to be called before LM calls.
func (m *CallbackManager) AddPreLMCallback(cb LMCallback) {
	m.preLMCallbacks = append(m.preLMCallbacks, cb)
}

// AddPostLMCallback adds a callback to be called after LM calls.
func (m *CallbackManager) AddPostLMCallback(cb LMCallback) {
	m.postLMCallbacks = append(m.postLMCallbacks, cb)
}

// AddPreModuleCallback adds a callback to be called before module execution.
func (m *CallbackManager) AddPreModuleCallback(cb ModuleCallback) {
	m.preModuleCallbacks = append(m.preModuleCallbacks, cb)
}

// AddPostModuleCallback adds a callback to be called after module execution.
func (m *CallbackManager) AddPostModuleCallback(cb ModuleCallback) {
	m.postModuleCallbacks = append(m.postModuleCallbacks, cb)
}

// OnPreLM triggers all pre-LM callbacks.
func (m *CallbackManager) OnPreLM(ctx context.Context, data LMCallbackData) {
	for _, cb := range m.preLMCallbacks {
		cb(ctx, data)
	}
}

// OnPostLM triggers all post-LM callbacks.
func (m *CallbackManager) OnPostLM(ctx context.Context, data LMCallbackData) {
	for _, cb := range m.postLMCallbacks {
		cb(ctx, data)
	}
}

// OnPreModule triggers all pre-module callbacks.
func (m *CallbackManager) OnPreModule(ctx context.Context, data ModuleCallbackData) {
	for _, cb := range m.preModuleCallbacks {
		cb(ctx, data)
	}
}

// OnPostModule triggers all post-module callbacks.
func (m *CallbackManager) OnPostModule(ctx context.Context, data ModuleCallbackData) {
	for _, cb := range m.postModuleCallbacks {
		cb(ctx, data)
	}
}

// Common callback implementations

// LoggingCallback creates a callback that logs operations.
func LoggingCallback(logger Logger) LMCallback {
	return func(ctx context.Context, data LMCallbackData) {
		if data.Error != nil {
			logger.Error("LM call failed", "model", data.Model, "error", data.Error)
		} else {
			logger.Info("LM call completed", "model", data.Model, "tokens", data.Tokens, "duration", data.Duration)
		}
	}
}

// MetricsCallback creates a callback that records metrics.
func MetricsCallback(recorder MetricsRecorder) LMCallback {
	return func(ctx context.Context, data LMCallbackData) {
		recorder.RecordLMCall(data.Model, data.Duration, data.Tokens, data.Error)
	}
}

// Logger is a simple logging interface.
type Logger interface {
	Info(msg string, args ...interface{})
	Error(msg string, args ...interface{})
}

// MetricsRecorder is an interface for recording metrics.
type MetricsRecorder interface {
	RecordLMCall(model string, duration time.Duration, tokens int, err error)
}

// ChainCallbacks chains multiple callbacks into one.
func ChainCallbacks(callbacks ...LMCallback) LMCallback {
	return func(ctx context.Context, data LMCallbackData) {
		for _, cb := range callbacks {
			cb(ctx, data)
		}
	}
}

// AsyncCallback wraps a callback to run asynchronously.
func AsyncCallback(cb LMCallback) LMCallback {
	return func(ctx context.Context, data LMCallbackData) {
		go cb(ctx, data)
	}
}
