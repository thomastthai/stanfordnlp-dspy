package utils

import (
	"context"
	"sync"
)

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
