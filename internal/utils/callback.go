package utils

import (
	"context"
	"sync"
)

// CallbackHandler defines the interface for callback handlers.
type CallbackHandler interface {
	// OnModuleStart is called when a module starts execution
	OnModuleStart(ctx context.Context, callID string, module string, inputs map[string]interface{})
	
	// OnModuleEnd is called when a module finishes execution
	OnModuleEnd(ctx context.Context, callID string, outputs map[string]interface{}, err error)
	
	// OnLMStart is called when a language model call starts
	OnLMStart(ctx context.Context, callID string, model string, inputs map[string]interface{})
	
	// OnLMEnd is called when a language model call ends
	OnLMEnd(ctx context.Context, callID string, outputs map[string]interface{}, err error)
	
	// OnRetrieverStart is called when a retriever call starts
	OnRetrieverStart(ctx context.Context, callID string, retriever string, query string)
	
	// OnRetrieverEnd is called when a retriever call ends
	OnRetrieverEnd(ctx context.Context, callID string, documents []string, err error)
	
	// OnToolStart is called when a tool execution starts
	OnToolStart(ctx context.Context, callID string, tool string, inputs map[string]interface{})
	
	// OnToolEnd is called when a tool execution ends
	OnToolEnd(ctx context.Context, callID string, outputs map[string]interface{}, err error)
}

// BaseCallback provides a default implementation of CallbackHandler.
// Users can embed this and override only the methods they need.
type BaseCallback struct{}

// OnModuleStart implements CallbackHandler.OnModuleStart.
func (b *BaseCallback) OnModuleStart(ctx context.Context, callID string, module string, inputs map[string]interface{}) {
}

// OnModuleEnd implements CallbackHandler.OnModuleEnd.
func (b *BaseCallback) OnModuleEnd(ctx context.Context, callID string, outputs map[string]interface{}, err error) {
}

// OnLMStart implements CallbackHandler.OnLMStart.
func (b *BaseCallback) OnLMStart(ctx context.Context, callID string, model string, inputs map[string]interface{}) {
}

// OnLMEnd implements CallbackHandler.OnLMEnd.
func (b *BaseCallback) OnLMEnd(ctx context.Context, callID string, outputs map[string]interface{}, err error) {
}

// OnRetrieverStart implements CallbackHandler.OnRetrieverStart.
func (b *BaseCallback) OnRetrieverStart(ctx context.Context, callID string, retriever string, query string) {
}

// OnRetrieverEnd implements CallbackHandler.OnRetrieverEnd.
func (b *BaseCallback) OnRetrieverEnd(ctx context.Context, callID string, documents []string, err error) {
}

// OnToolStart implements CallbackHandler.OnToolStart.
func (b *BaseCallback) OnToolStart(ctx context.Context, callID string, tool string, inputs map[string]interface{}) {
}

// OnToolEnd implements CallbackHandler.OnToolEnd.
func (b *BaseCallback) OnToolEnd(ctx context.Context, callID string, outputs map[string]interface{}, err error) {
}

// CallbackManager manages multiple callback handlers.
type CallbackManager struct {
	handlers []CallbackHandler
	mu       sync.RWMutex
// Callback represents a callback function that can be invoked at various points.
type Callback interface {
	// OnStart is called when an operation starts.
	OnStart(ctx context.Context, inputs map[string]interface{}) error

	// OnEnd is called when an operation completes successfully.
	OnEnd(ctx context.Context, outputs map[string]interface{}) error

	// OnError is called when an operation fails.
	OnError(ctx context.Context, err error) error
}

// CallbackManager manages multiple callbacks.
type CallbackManager struct {
	callbacks []Callback
	mu        sync.RWMutex
}

// NewCallbackManager creates a new callback manager.
func NewCallbackManager() *CallbackManager {
	return &CallbackManager{
		handlers: make([]CallbackHandler, 0),
	}
}

// AddHandler adds a callback handler.
func (cm *CallbackManager) AddHandler(handler CallbackHandler) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.handlers = append(cm.handlers, handler)
}

// RemoveHandler removes a callback handler.
func (cm *CallbackManager) RemoveHandler(handler CallbackHandler) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	for i, h := range cm.handlers {
		if h == handler {
			cm.handlers = append(cm.handlers[:i], cm.handlers[i+1:]...)
		callbacks: make([]Callback, 0),
	}
}

// Add adds a callback to the manager.
func (cm *CallbackManager) Add(callback Callback) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.callbacks = append(cm.callbacks, callback)
}

// Remove removes a callback from the manager.
func (cm *CallbackManager) Remove(callback Callback) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	for i, cb := range cm.callbacks {
		if cb == callback {
			cm.callbacks = append(cm.callbacks[:i], cm.callbacks[i+1:]...)
			break
		}
	}
}

// OnModuleStart notifies all handlers about module start.
func (cm *CallbackManager) OnModuleStart(ctx context.Context, callID string, module string, inputs map[string]interface{}) {
	cm.mu.RLock()
	handlers := make([]CallbackHandler, len(cm.handlers))
	copy(handlers, cm.handlers)
	cm.mu.RUnlock()
	
	for _, handler := range handlers {
		handler.OnModuleStart(ctx, callID, module, inputs)
	}
}

// OnModuleEnd notifies all handlers about module end.
func (cm *CallbackManager) OnModuleEnd(ctx context.Context, callID string, outputs map[string]interface{}, err error) {
	cm.mu.RLock()
	handlers := make([]CallbackHandler, len(cm.handlers))
	copy(handlers, cm.handlers)
	cm.mu.RUnlock()
	
	for _, handler := range handlers {
		handler.OnModuleEnd(ctx, callID, outputs, err)
	}
}

// OnLMStart notifies all handlers about LM start.
func (cm *CallbackManager) OnLMStart(ctx context.Context, callID string, model string, inputs map[string]interface{}) {
	cm.mu.RLock()
	handlers := make([]CallbackHandler, len(cm.handlers))
	copy(handlers, cm.handlers)
	cm.mu.RUnlock()
	
	for _, handler := range handlers {
		handler.OnLMStart(ctx, callID, model, inputs)
	}
}

// OnLMEnd notifies all handlers about LM end.
func (cm *CallbackManager) OnLMEnd(ctx context.Context, callID string, outputs map[string]interface{}, err error) {
	cm.mu.RLock()
	handlers := make([]CallbackHandler, len(cm.handlers))
	copy(handlers, cm.handlers)
	cm.mu.RUnlock()
	
	for _, handler := range handlers {
		handler.OnLMEnd(ctx, callID, outputs, err)
	}
}

// OnRetrieverStart notifies all handlers about retriever start.
func (cm *CallbackManager) OnRetrieverStart(ctx context.Context, callID string, retriever string, query string) {
	cm.mu.RLock()
	handlers := make([]CallbackHandler, len(cm.handlers))
	copy(handlers, cm.handlers)
	cm.mu.RUnlock()
	
	for _, handler := range handlers {
		handler.OnRetrieverStart(ctx, callID, retriever, query)
	}
}

// OnRetrieverEnd notifies all handlers about retriever end.
func (cm *CallbackManager) OnRetrieverEnd(ctx context.Context, callID string, documents []string, err error) {
	cm.mu.RLock()
	handlers := make([]CallbackHandler, len(cm.handlers))
	copy(handlers, cm.handlers)
	cm.mu.RUnlock()
	
	for _, handler := range handlers {
		handler.OnRetrieverEnd(ctx, callID, documents, err)
	}
}

// OnToolStart notifies all handlers about tool start.
func (cm *CallbackManager) OnToolStart(ctx context.Context, callID string, tool string, inputs map[string]interface{}) {
	cm.mu.RLock()
	handlers := make([]CallbackHandler, len(cm.handlers))
	copy(handlers, cm.handlers)
	cm.mu.RUnlock()
	
	for _, handler := range handlers {
		handler.OnToolStart(ctx, callID, tool, inputs)
	}
}

// OnToolEnd notifies all handlers about tool end.
func (cm *CallbackManager) OnToolEnd(ctx context.Context, callID string, outputs map[string]interface{}, err error) {
	cm.mu.RLock()
	handlers := make([]CallbackHandler, len(cm.handlers))
	copy(handlers, cm.handlers)
	cm.mu.RUnlock()
	
	for _, handler := range handlers {
		handler.OnToolEnd(ctx, callID, outputs, err)
	}
}

// LoggingCallback is a simple callback that logs all events.
type LoggingCallback struct {
	BaseCallback
	logger func(format string, args ...interface{})
}

// NewLoggingCallback creates a new logging callback.
func NewLoggingCallback(logger func(format string, args ...interface{})) *LoggingCallback {
	return &LoggingCallback{logger: logger}
}

// OnModuleStart logs module start.
func (lc *LoggingCallback) OnModuleStart(ctx context.Context, callID string, module string, inputs map[string]interface{}) {
	lc.logger("Module %s started (call_id=%s)", module, callID)
}

// OnModuleEnd logs module end.
func (lc *LoggingCallback) OnModuleEnd(ctx context.Context, callID string, outputs map[string]interface{}, err error) {
	if err != nil {
		lc.logger("Module ended with error (call_id=%s): %v", callID, err)
	} else {
		lc.logger("Module ended successfully (call_id=%s)", callID)
	}
}

// OnLMStart logs LM start.
func (lc *LoggingCallback) OnLMStart(ctx context.Context, callID string, model string, inputs map[string]interface{}) {
	lc.logger("LM %s called (call_id=%s)", model, callID)
}

// OnLMEnd logs LM end.
func (lc *LoggingCallback) OnLMEnd(ctx context.Context, callID string, outputs map[string]interface{}, err error) {
	if err != nil {
		lc.logger("LM call ended with error (call_id=%s): %v", callID, err)
	} else {
		lc.logger("LM call ended successfully (call_id=%s)", callID)
	}
// OnStart calls all OnStart callbacks.
func (cm *CallbackManager) OnStart(ctx context.Context, inputs map[string]interface{}) error {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	for _, callback := range cm.callbacks {
		if err := callback.OnStart(ctx, inputs); err != nil {
			return err
		}
	}
	return nil
}

// OnEnd calls all OnEnd callbacks.
func (cm *CallbackManager) OnEnd(ctx context.Context, outputs map[string]interface{}) error {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	for _, callback := range cm.callbacks {
		if err := callback.OnEnd(ctx, outputs); err != nil {
			return err
		}
	}
	return nil
}

// OnError calls all OnError callbacks.
func (cm *CallbackManager) OnError(ctx context.Context, err error) error {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	for _, callback := range cm.callbacks {
		if callbackErr := callback.OnError(ctx, err); callbackErr != nil {
			return callbackErr
		}
	}
	return nil
}

// SimpleCallback is a basic callback implementation using function pointers.
type SimpleCallback struct {
	OnStartFunc func(ctx context.Context, inputs map[string]interface{}) error
	OnEndFunc   func(ctx context.Context, outputs map[string]interface{}) error
	OnErrorFunc func(ctx context.Context, err error) error
}

// OnStart implements Callback.OnStart.
func (sc *SimpleCallback) OnStart(ctx context.Context, inputs map[string]interface{}) error {
	if sc.OnStartFunc != nil {
		return sc.OnStartFunc(ctx, inputs)
	}
	return nil
}

// OnEnd implements Callback.OnEnd.
func (sc *SimpleCallback) OnEnd(ctx context.Context, outputs map[string]interface{}) error {
	if sc.OnEndFunc != nil {
		return sc.OnEndFunc(ctx, outputs)
	}
	return nil
}

// OnError implements Callback.OnError.
func (sc *SimpleCallback) OnError(ctx context.Context, err error) error {
	if sc.OnErrorFunc != nil {
		return sc.OnErrorFunc(ctx, err)
	}
	return nil
}
