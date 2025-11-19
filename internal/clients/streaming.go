// Package clients provides shared streaming utilities for LM providers.
package clients

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"strings"
)

// StreamChunk represents a chunk of streaming response.
type StreamChunk struct {
	// Delta is the incremental content
	Delta string

	// IsError indicates if this chunk contains an error
	IsError bool

	// Error contains the error if IsError is true
	Error error

	// Done indicates if the stream has completed
	Done bool

	// Metadata contains additional chunk-specific data
	Metadata map[string]interface{}
}

// SSEReader reads Server-Sent Events (SSE) from a stream.
type SSEReader struct {
	reader  *bufio.Reader
	scanner *bufio.Scanner
}

// NewSSEReader creates a new SSE reader.
func NewSSEReader(r io.Reader) *SSEReader {
	scanner := bufio.NewScanner(r)
	return &SSEReader{
		reader:  bufio.NewReader(r),
		scanner: scanner,
	}
}

// ReadEvent reads the next SSE event from the stream.
// It returns the event data or an error if reading fails.
func (s *SSEReader) ReadEvent() ([]byte, error) {
	if !s.scanner.Scan() {
		if err := s.scanner.Err(); err != nil {
			return nil, fmt.Errorf("error reading stream: %w", err)
		}
		return nil, io.EOF
	}

	line := s.scanner.Text()

	// SSE lines start with "data: "
	if !strings.HasPrefix(line, "data: ") {
		// Skip non-data lines (e.g., comments, event types)
		return s.ReadEvent()
	}

	data := strings.TrimPrefix(line, "data: ")

	// Check for stream end marker
	if data == "[DONE]" {
		return nil, io.EOF
	}

	return []byte(data), nil
}

// StreamToChannel converts SSE events to a channel of StreamChunk.
func StreamToChannel(ctx context.Context, reader io.Reader, parser func([]byte) (*StreamChunk, error)) (<-chan StreamChunk, <-chan error) {
	chunkCh := make(chan StreamChunk, 10)
	errCh := make(chan error, 1)

	go func() {
		defer close(chunkCh)
		defer close(errCh)

		sseReader := NewSSEReader(reader)

		for {
			select {
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			default:
			}

			data, err := sseReader.ReadEvent()
			if err != nil {
				if err == io.EOF {
					// Stream ended normally
					return
				}
				errCh <- fmt.Errorf("error reading event: %w", err)
				return
			}

			// Parse the chunk
			chunk, err := parser(data)
			if err != nil {
				// Skip malformed chunks but continue
				continue
			}

			// Send chunk
			select {
			case chunkCh <- *chunk:
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}

			// Check if this is the final chunk
			if chunk.Done {
				return
			}
		}
	}()

	return chunkCh, errCh
}

// CollectStreamText collects all text from a stream channel.
func CollectStreamText(chunkCh <-chan StreamChunk, errCh <-chan error) (string, error) {
	var builder strings.Builder

	for {
		select {
		case chunk, ok := <-chunkCh:
			if !ok {
				// Channel closed, check for errors
				select {
				case err := <-errCh:
					if err != nil {
						return "", err
					}
				default:
				}
				return builder.String(), nil
			}

			if chunk.IsError {
				return "", chunk.Error
			}

			if chunk.Delta != "" {
				builder.WriteString(chunk.Delta)
			}

		case err := <-errCh:
			if err != nil {
				return "", err
			}
		}
	}
}
