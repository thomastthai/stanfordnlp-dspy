package utils

import (
	"context"
	"sync"
)

// AsyncFunc represents an asynchronous function that can be executed concurrently.
type AsyncFunc func(ctx context.Context) (interface{}, error)

// WorkerPool manages a pool of goroutines for concurrent execution.
type WorkerPool struct {
	maxWorkers int
	taskQueue  chan AsyncFunc
	results    chan AsyncResult
	ctx        context.Context
	cancel     context.CancelFunc
	wg         sync.WaitGroup
}

// AsyncResult represents the result of an asynchronous operation.
type AsyncResult struct {
	Value interface{}
	Error error
	Index int
}

// NewWorkerPool creates a new worker pool with the specified number of workers.
func NewWorkerPool(ctx context.Context, maxWorkers int) *WorkerPool {
	if maxWorkers <= 0 {
		maxWorkers = 1
	}

	poolCtx, cancel := context.WithCancel(ctx)

	pool := &WorkerPool{
		maxWorkers: maxWorkers,
		taskQueue:  make(chan AsyncFunc, maxWorkers*2),
		results:    make(chan AsyncResult, maxWorkers*2),
		ctx:        poolCtx,
		cancel:     cancel,
	}

	// Start workers
	for i := 0; i < maxWorkers; i++ {
		pool.wg.Add(1)
		go pool.worker()
	}

	return pool
}

// worker is the goroutine that processes tasks from the queue.
func (wp *WorkerPool) worker() {
	defer wp.wg.Done()

	for {
		select {
		case <-wp.ctx.Done():
			return
		case task, ok := <-wp.taskQueue:
			if !ok {
				return
			}

			result, err := task(wp.ctx)
			select {
			case wp.results <- AsyncResult{Value: result, Error: err}:
			case <-wp.ctx.Done():
				return
			}
		}
	}
}

// Submit submits a task to the worker pool.
func (wp *WorkerPool) Submit(task AsyncFunc) {
	select {
	case wp.taskQueue <- task:
	case <-wp.ctx.Done():
	}
}

// Results returns the results channel.
func (wp *WorkerPool) Results() <-chan AsyncResult {
	return wp.results
}

// Close closes the worker pool and waits for all workers to finish.
func (wp *WorkerPool) Close() {
	close(wp.taskQueue)
	wp.wg.Wait()
	close(wp.results)
	wp.cancel()
}

// Asyncify converts a synchronous function to run asynchronously.
func Asyncify(ctx context.Context, fn func() (interface{}, error)) <-chan AsyncResult {
	result := make(chan AsyncResult, 1)

	go func() {
		defer close(result)

		value, err := fn()
		select {
		case result <- AsyncResult{Value: value, Error: err}:
		case <-ctx.Done():
			result <- AsyncResult{Error: ctx.Err()}
		}
	}()

	return result
}

// AsyncifyBatch runs multiple functions asynchronously with a worker pool.
func AsyncifyBatch(ctx context.Context, fns []func() (interface{}, error), maxWorkers int) []AsyncResult {
	if len(fns) == 0 {
		return []AsyncResult{}
	}

	pool := NewWorkerPool(ctx, maxWorkers)
	defer pool.Close()

	// Submit all tasks
	for i, fn := range fns {
		idx := i
		f := fn
		pool.Submit(func(ctx context.Context) (interface{}, error) {
			result, err := f()
			if err != nil {
				return AsyncResult{Value: result, Error: err, Index: idx}, nil
			}
			return AsyncResult{Value: result, Error: nil, Index: idx}, nil
		})
	}

	// Collect results
	results := make([]AsyncResult, len(fns))
	for i := 0; i < len(fns); i++ {
		select {
		case res := <-pool.Results():
			if ar, ok := res.Value.(AsyncResult); ok {
				results[ar.Index] = ar
			} else {
				results[i] = res
			}
		case <-ctx.Done():
			for j := i; j < len(fns); j++ {
				results[j] = AsyncResult{Error: ctx.Err()}
			}
			return results
		}
	}

	return results
}

// Parallel runs multiple functions in parallel and returns their results.
func Parallel(ctx context.Context, fns ...func() (interface{}, error)) []AsyncResult {
	if len(fns) == 0 {
		return []AsyncResult{}
	}

	results := make([]AsyncResult, len(fns))
	var wg sync.WaitGroup

	for i, fn := range fns {
		wg.Add(1)
		go func(idx int, f func() (interface{}, error)) {
			defer wg.Done()

			value, err := f()
			results[idx] = AsyncResult{Value: value, Error: err, Index: idx}
		}(i, fn)
	}

	wg.Wait()
	return results
}
