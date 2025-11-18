package streaming

import (
	"context"
)

// Streamify wraps a function to make it streaming-enabled.
// It takes a function that produces output and converts it to stream tokens incrementally.
type Streamify[T any] struct {
	bufferSize int
}

// NewStreamify creates a new Streamify instance.
func NewStreamify[T any](bufferSize int) *Streamify[T] {
	if bufferSize <= 0 {
		bufferSize = 1
	}
	return &Streamify[T]{
		bufferSize: bufferSize,
	}
}

// Wrap converts a synchronous function into a streaming version.
// The wrapped function returns a stream that can be consumed incrementally.
func (s *Streamify[T]) Wrap(fn func(context.Context) (T, error)) func(context.Context) (*Stream[T], error) {
	return func(ctx context.Context) (*Stream[T], error) {
		stream := NewStream[T](ctx, s.bufferSize)

		go func() {
			defer stream.Close()

			result, err := fn(ctx)
			if err != nil {
				// In case of error, close the stream
				return
			}

			// Send the result to the stream
			stream.Send(result)
		}()

		return stream, nil
	}
}

// StreamTokens streams tokens from a string output incrementally.
// This is useful for streaming text generation outputs token by token.
func StreamTokens(ctx context.Context, text string) *Stream[string] {
	stream := NewStream[string](ctx, 0)

	go func() {
		defer stream.Close()

		// For simplicity, we split by spaces to simulate token streaming
		// A real implementation would use a proper tokenizer
		tokens := tokenize(text)
		for _, token := range tokens {
			select {
			case <-ctx.Done():
				return
			default:
				if err := stream.Send(token); err != nil {
					return
				}
			}
		}
	}()

	return stream
}

// tokenize is a simple tokenizer that splits text by spaces.
// A real implementation would use a proper tokenizer like tiktoken.
func tokenize(text string) []string {
	if text == "" {
		return nil
	}

	var tokens []string
	var current string

	for _, ch := range text {
		if ch == ' ' || ch == '\n' || ch == '\t' {
			if current != "" {
				tokens = append(tokens, current)
				current = ""
			}
			tokens = append(tokens, string(ch))
		} else {
			current += string(ch)
		}
	}

	if current != "" {
		tokens = append(tokens, current)
	}

	return tokens
}
