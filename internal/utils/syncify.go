package utils

import (
	"context"
	"fmt"
	"time"
)

// Syncify converts an asynchronous function to a synchronous one with timeout.
func Syncify(ctx context.Context, asyncFn func(context.Context) <-chan AsyncResult, timeout time.Duration) (interface{}, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	resultChan := asyncFn(timeoutCtx)

	select {
	case result := <-resultChan:
		return result.Value, result.Error
	case <-timeoutCtx.Done():
		return nil, fmt.Errorf("operation timed out after %v", timeout)
	}
}

// SyncifyWithDefault converts an async function to sync with a default value on timeout/error.
func SyncifyWithDefault(ctx context.Context, asyncFn func(context.Context) <-chan AsyncResult, timeout time.Duration, defaultValue interface{}) interface{} {
	value, err := Syncify(ctx, asyncFn, timeout)
	if err != nil {
		return defaultValue
	}
	return value
}

// WaitFor waits for a channel to produce a value or times out.
func WaitFor(ctx context.Context, ch <-chan interface{}, timeout time.Duration) (interface{}, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	select {
	case value := <-ch:
		return value, nil
	case <-timeoutCtx.Done():
		if timeoutCtx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("operation timed out after %v", timeout)
		}
		return nil, timeoutCtx.Err()
	}
}

// WaitAll waits for all channels to complete or times out.
func WaitAll(ctx context.Context, channels []<-chan interface{}, timeout time.Duration) ([]interface{}, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	results := make([]interface{}, len(channels))

	for i, ch := range channels {
		select {
		case value := <-ch:
			results[i] = value
		case <-timeoutCtx.Done():
			if timeoutCtx.Err() == context.DeadlineExceeded {
				return results, fmt.Errorf("operation timed out after %v", timeout)
			}
			return results, timeoutCtx.Err()
		}
	}

	return results, nil
}

// WaitAny waits for any of the channels to produce a value or times out.
func WaitAny(ctx context.Context, channels []<-chan interface{}, timeout time.Duration) (interface{}, int, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Create a single channel that receives from all input channels
	merged := make(chan struct {
		value interface{}
		index int
	}, len(channels))

	for i, ch := range channels {
		idx := i
		c := ch
		go func() {
			select {
			case value := <-c:
				merged <- struct {
					value interface{}
					index int
				}{value: value, index: idx}
			case <-timeoutCtx.Done():
			}
		}()
	}

	select {
	case result := <-merged:
		return result.value, result.index, nil
	case <-timeoutCtx.Done():
		if timeoutCtx.Err() == context.DeadlineExceeded {
			return nil, -1, fmt.Errorf("operation timed out after %v", timeout)
		}
		return nil, -1, timeoutCtx.Err()
	}
}

// BlockingWrapper wraps an async function to make it blocking with timeout.
type BlockingWrapper struct {
	timeout time.Duration
}

// NewBlockingWrapper creates a new blocking wrapper with the specified timeout.
func NewBlockingWrapper(timeout time.Duration) *BlockingWrapper {
	return &BlockingWrapper{timeout: timeout}
}

// Call calls an async function and blocks until it completes or times out.
func (bw *BlockingWrapper) Call(ctx context.Context, asyncFn func(context.Context) <-chan AsyncResult) (interface{}, error) {
	return Syncify(ctx, asyncFn, bw.timeout)
}

// CallWithDefault calls an async function and returns a default value on error/timeout.
func (bw *BlockingWrapper) CallWithDefault(ctx context.Context, asyncFn func(context.Context) <-chan AsyncResult, defaultValue interface{}) interface{} {
	return SyncifyWithDefault(ctx, asyncFn, bw.timeout, defaultValue)
}
