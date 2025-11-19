package clients

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestSSEReader(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []string
		wantErr  bool
	}{
		{
			name: "simple events",
			input: `data: {"content": "Hello"}
data: {"content": "World"}
data: [DONE]
`,
			expected: []string{
				`{"content": "Hello"}`,
				`{"content": "World"}`,
			},
			wantErr: false,
		},
		{
			name: "events with comments",
			input: `:comment
data: {"content": "Test"}
:another comment
data: [DONE]
`,
			expected: []string{
				`{"content": "Test"}`,
			},
			wantErr: false,
		},
		{
			name: "empty stream",
			input: `data: [DONE]
`,
			expected: []string{},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := NewSSEReader(strings.NewReader(tt.input))
			var results []string

			for {
				data, err := reader.ReadEvent()
				if err != nil {
					break
				}
				results = append(results, string(data))
			}

			if len(results) != len(tt.expected) {
				t.Errorf("got %d events, want %d", len(results), len(tt.expected))
				return
			}

			for i, result := range results {
				if result != tt.expected[i] {
					t.Errorf("event %d: got %q, want %q", i, result, tt.expected[i])
				}
			}
		})
	}
}

func TestStreamToChannel(t *testing.T) {
	input := `data: {"delta": "Hello"}
data: {"delta": " World"}
data: [DONE]
`

	parser := func(data []byte) (*StreamChunk, error) {
		return &StreamChunk{
			Delta: string(data),
			Done:  false,
		}, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	chunkCh, errCh := StreamToChannel(ctx, strings.NewReader(input), parser)

	var chunks []string
	for {
		select {
		case chunk, ok := <-chunkCh:
			if !ok {
				// Channel closed
				select {
				case err := <-errCh:
					if err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
				default:
				}
				goto done
			}
			chunks = append(chunks, chunk.Delta)

		case err := <-errCh:
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		case <-ctx.Done():
			t.Fatal("timeout waiting for stream")
		}
	}

done:
	if len(chunks) != 2 {
		t.Errorf("got %d chunks, want 2", len(chunks))
	}
}

func TestCollectStreamText(t *testing.T) {
	chunkCh := make(chan StreamChunk, 3)
	errCh := make(chan error, 1)

	// Send some chunks
	chunkCh <- StreamChunk{Delta: "Hello"}
	chunkCh <- StreamChunk{Delta: " "}
	chunkCh <- StreamChunk{Delta: "World"}
	close(chunkCh)
	close(errCh)

	text, err := CollectStreamText(chunkCh, errCh)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := "Hello World"
	if text != expected {
		t.Errorf("got %q, want %q", text, expected)
	}
}

func TestCollectStreamTextWithError(t *testing.T) {
	chunkCh := make(chan StreamChunk, 1)
	errCh := make(chan error, 1)

	// Send an error chunk
	chunkCh <- StreamChunk{
		IsError: true,
		Error:   &ClientError{Message: "test error"},
	}
	close(chunkCh)
	close(errCh)

	_, err := CollectStreamText(chunkCh, errCh)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}
