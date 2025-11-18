package utils

import (
	"context"
)

// Syncify provides utilities for converting asynchronous operations to synchronous.
type Syncify struct{}

// NewSyncify creates a new Syncify instance.
func NewSyncify() *Syncify {
	return &Syncify{}
}

// Await waits for an async channel to complete and returns the result synchronously.
func (s *Syncify) Await(ctx context.Context, resultCh <-chan Result) (interface{}, error) {
	return Await(ctx, resultCh)
}

// AwaitAll waits for all async channels to complete and returns the results synchronously.
func (s *Syncify) AwaitAll(ctx context.Context, resultsCh <-chan []Result) ([]interface{}, []error) {
	return AwaitAll(ctx, resultsCh)
}

// Run executes an async function synchronously by waiting for it to complete.
func (s *Syncify) Run(ctx context.Context, fn func(context.Context) (interface{}, error)) (interface{}, error) {
	return fn(ctx)
}
