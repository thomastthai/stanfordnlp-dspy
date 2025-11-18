// Package streaming provides streaming utilities for DSPy modules.
package streaming

import (
	"context"
)

// Stream represents a streaming output channel.
type Stream[T any] struct {
	ch     chan T
	ctx    context.Context
	cancel context.CancelFunc
}

// NewStream creates a new stream with the given buffer size.
func NewStream[T any](ctx context.Context, bufferSize int) *Stream[T] {
	if bufferSize <= 0 {
		bufferSize = 1
	}

	streamCtx, cancel := context.WithCancel(ctx)
	return &Stream[T]{
		ch:     make(chan T, bufferSize),
		ctx:    streamCtx,
		cancel: cancel,
	}
}

// Send sends a value to the stream.
// Returns an error if the context is cancelled or the stream is closed.
func (s *Stream[T]) Send(value T) error {
	select {
	case <-s.ctx.Done():
		return s.ctx.Err()
	case s.ch <- value:
		return nil
	}
}

// Receive receives the next value from the stream.
// Returns false if the stream is closed.
func (s *Stream[T]) Receive() (T, bool) {
	select {
	case <-s.ctx.Done():
		var zero T
		return zero, false
	case value, ok := <-s.ch:
		return value, ok
	}
}

// Chan returns the underlying channel for range iteration.
func (s *Stream[T]) Chan() <-chan T {
	return s.ch
}

// Close closes the stream without cancelling the context.
func (s *Stream[T]) Close() {
	close(s.ch)
}

// Context returns the stream's context.
func (s *Stream[T]) Context() context.Context {
	return s.ctx
}

// Filter creates a new stream that only includes values that match the predicate.
func Filter[T any](ctx context.Context, input *Stream[T], predicate func(T) bool) *Stream[T] {
	output := NewStream[T](ctx, 0)

	go func() {
		defer output.Close()
		for {
			value, ok := input.Receive()
			if !ok {
				return
			}
			if predicate(value) {
				if err := output.Send(value); err != nil {
					return
				}
			}
		}
	}()

	return output
}

// Map creates a new stream that applies a transformation to each value.
func Map[T, U any](ctx context.Context, input *Stream[T], mapper func(T) U) *Stream[U] {
	output := NewStream[U](ctx, 0)

	go func() {
		defer output.Close()
		for {
			value, ok := input.Receive()
			if !ok {
				return
			}
			mapped := mapper(value)
			if err := output.Send(mapped); err != nil {
				return
			}
		}
	}()

	return output
}

// Aggregate collects all values from a stream into a slice.
func Aggregate[T any](ctx context.Context, input *Stream[T]) ([]T, error) {
	var results []T

	for {
		select {
		case <-ctx.Done():
			return results, ctx.Err()
		case value, ok := <-input.Chan():
			if !ok {
				return results, nil
			}
			results = append(results, value)
		}
	}
}

// Buffer creates a new stream that buffers values up to maxSize before sending.
func Buffer[T any](ctx context.Context, input *Stream[T], maxSize int) *Stream[[]T] {
	output := NewStream[[]T](ctx, 0)

	go func() {
		defer output.Close()
		var buffer []T

		for {
			value, ok := input.Receive()
			if !ok {
				// Send remaining buffer if any
				if len(buffer) > 0 {
					output.Send(buffer)
				}
				return
			}

			buffer = append(buffer, value)
			if len(buffer) >= maxSize {
				if err := output.Send(buffer); err != nil {
					return
				}
				buffer = nil
			}
		}
	}()

	return output
}
