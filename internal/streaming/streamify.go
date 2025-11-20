package streaming

import (
	"context"
	"fmt"
)

// StreamifyFunc wraps a function to make it return a stream.
type StreamifyFunc func(ctx context.Context, input interface{}) *Stream

// Streamify wraps a synchronous function to make it stream outputs.
// This is useful for converting batch processing to streaming.
func Streamify(fn func(interface{}) ([]interface{}, error)) StreamifyFunc {
	return func(ctx context.Context, input interface{}) *Stream {
		stream := NewStream(ctx, 10)

		go func() {
			defer stream.Close()

			results, err := fn(input)
			if err != nil {
				stream.SendError(err)
				return
			}

			for _, result := range results {
				if err := stream.Send(result); err != nil {
					return
				}
			}
		}()

		return stream
	}
}

// StreamifyChunked wraps a function to stream outputs in chunks.
func StreamifyChunked(fn func(interface{}) ([]interface{}, error), chunkSize int) StreamifyFunc {
	return func(ctx context.Context, input interface{}) *Stream {
		stream := NewStream(ctx, 10)

		go func() {
			defer stream.Close()

			results, err := fn(input)
			if err != nil {
				stream.SendError(err)
				return
			}

			// Send in chunks
			for i := 0; i < len(results); i += chunkSize {
				end := i + chunkSize
				if end > len(results) {
					end = len(results)
				}

				chunk := results[i:end]
				if err := stream.Send(chunk); err != nil {
					return
				}
			}
		}()

		return stream
	}
}

// StreamifyModule wraps a module's forward function for streaming.
type Module interface {
	Forward(context.Context, map[string]interface{}) (map[string]interface{}, error)
}

// StreamifyModuleFunc creates a streaming version of a module.
func StreamifyModuleFunc(module Module) func(context.Context, map[string]interface{}) *Stream {
	return func(ctx context.Context, inputs map[string]interface{}) *Stream {
		stream := NewStream(ctx, 10)

		go func() {
			defer stream.Close()

			// Send start event
			stream.Send(map[string]interface{}{
				"event": "start",
				"input": inputs,
			})

			// Call module
			outputs, err := module.Forward(ctx, inputs)
			if err != nil {
				stream.SendError(err)
				return
			}

			// Send output event
			stream.Send(map[string]interface{}{
				"event":  "output",
				"result": outputs,
			})

			// Send end event
			stream.Send(map[string]interface{}{
				"event": "end",
			})
		}()

		return stream
	}
}

// StreamBuffer buffers stream items to handle backpressure.
type StreamBuffer struct {
	input      *Stream
	output     *Stream
	bufferSize int
}

// NewStreamBuffer creates a new stream buffer.
func NewStreamBuffer(ctx context.Context, input *Stream, bufferSize int) *StreamBuffer {
	output := NewStream(ctx, bufferSize)

	buffer := &StreamBuffer{
		input:      input,
		output:     output,
		bufferSize: bufferSize,
	}

	go buffer.process()

	return buffer
}

// process processes items from input to output with buffering.
func (sb *StreamBuffer) process() {
	defer sb.output.Close()

	buffer := make([]interface{}, 0, sb.bufferSize)

	for {
		item, err := sb.input.Next()
		if err != nil {
			if len(buffer) > 0 {
				sb.flush(buffer)
			}
			if err.Error() != "EOF" {
				sb.output.SendError(err)
			}
			break
		}

		buffer = append(buffer, item)

		if len(buffer) >= sb.bufferSize {
			sb.flush(buffer)
			buffer = make([]interface{}, 0, sb.bufferSize)
		}
	}
}

// flush flushes the buffer to the output stream.
func (sb *StreamBuffer) flush(buffer []interface{}) {
	for _, item := range buffer {
		sb.output.Send(item)
	}
}

// Output returns the output stream.
func (sb *StreamBuffer) Output() *Stream {
	return sb.output
}

// StreamingListener listens to stream events.
type StreamingListener struct {
	OnItem  func(interface{})
	OnError func(error)
	OnDone  func()
}

// Listen starts listening to a stream with the provided callbacks.
func (sl *StreamingListener) Listen(stream *Stream) {
	go func() {
		for {
			item, err := stream.Next()
			if err != nil {
				if err.Error() == "EOF" {
					if sl.OnDone != nil {
						sl.OnDone()
					}
				} else if sl.OnError != nil {
					sl.OnError(err)
				}
				break
			}

			if sl.OnItem != nil {
				sl.OnItem(item)
			}
		}
	}()
}

// TokenStream represents a stream of tokens (for LM streaming).
type TokenStream struct {
	*Stream
	fullText string
}

// NewTokenStream creates a new token stream.
func NewTokenStream(ctx context.Context) *TokenStream {
	return &TokenStream{
		Stream:   NewStream(ctx, 100),
		fullText: "",
	}
}

// SendToken sends a token to the stream and accumulates it.
func (ts *TokenStream) SendToken(token string) error {
	ts.fullText += token
	return ts.Send(token)
}

// GetFullText returns the accumulated full text.
func (ts *TokenStream) GetFullText() string {
	return ts.fullText
}

// StreamCollector collects streaming output into different formats.
type StreamCollector struct {
	items  []interface{}
	errors []error
}

// NewStreamCollector creates a new stream collector.
func NewStreamCollector() *StreamCollector {
	return &StreamCollector{
		items:  make([]interface{}, 0),
		errors: make([]error, 0),
	}
}

// Collect collects items from a stream.
func (sc *StreamCollector) Collect(stream *Stream) error {
	for {
		item, err := stream.Next()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			sc.errors = append(sc.errors, err)
			return err
		}
		sc.items = append(sc.items, item)
	}
	return nil
}

// Items returns collected items.
func (sc *StreamCollector) Items() []interface{} {
	return sc.items
}

// Errors returns collected errors.
func (sc *StreamCollector) Errors() []error {
	return sc.errors
}

// HasErrors returns true if any errors were collected.
func (sc *StreamCollector) HasErrors() bool {
	return len(sc.errors) > 0
}

// GetItemsAsStrings converts items to strings.
func (sc *StreamCollector) GetItemsAsStrings() []string {
	strings := make([]string, len(sc.items))
	for i, item := range sc.items {
		strings[i] = fmt.Sprintf("%v", item)
	}
	return strings
}
