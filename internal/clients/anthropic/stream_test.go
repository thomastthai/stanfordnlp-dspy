package anthropic

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stanfordnlp/dspy/internal/clients"
)

func TestParseSSEStream(t *testing.T) {
	// Sample SSE stream from Anthropic API
	sseData := `event: message_start
data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant"}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" World"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}

event: message_stop
data: {"type":"message_stop"}
`

	client := &Client{}
	chunkCh := make(chan clients.StreamChunk, 10)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	go func() {
		err := client.parseSSEStream(ctx, strings.NewReader(sseData), chunkCh)
		if err != nil {
			t.Errorf("parseSSEStream error: %v", err)
		}
		close(chunkCh)
	}()

	var chunks []string
	for chunk := range chunkCh {
		if chunk.Delta != "" {
			chunks = append(chunks, chunk.Delta)
		}
	}

	// We should have collected "Hello" and " World"
	if len(chunks) != 2 {
		t.Errorf("expected 2 chunks, got %d", len(chunks))
	}

	if len(chunks) > 0 && chunks[0] != "Hello" {
		t.Errorf("first chunk: got %q, want %q", chunks[0], "Hello")
	}

	if len(chunks) > 1 && chunks[1] != " World" {
		t.Errorf("second chunk: got %q, want %q", chunks[1], " World")
	}
}

func TestStreamToText(t *testing.T) {
	chunkCh := make(chan clients.StreamChunk, 3)
	errCh := make(chan error, 1)

	// Send some chunks
	chunkCh <- clients.StreamChunk{Delta: "Hello"}
	chunkCh <- clients.StreamChunk{Delta: " "}
	chunkCh <- clients.StreamChunk{Delta: "World"}
	close(chunkCh)
	close(errCh)

	text, err := StreamToText(chunkCh, errCh)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := "Hello World"
	if text != expected {
		t.Errorf("got %q, want %q", text, expected)
	}
}

func TestStreamChunkParsing(t *testing.T) {
	tests := []struct {
		name     string
		jsonData string
		wantText string
		wantDone bool
	}{
		{
			name:     "text delta",
			jsonData: `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			wantText: "Hello",
			wantDone: false,
		},
		{
			name:     "message stop",
			jsonData: `{"type":"message_stop"}`,
			wantText: "",
			wantDone: true,
		},
		{
			name:     "message delta with stop",
			jsonData: `{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
			wantText: "",
			wantDone: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var streamChunk StreamChunk
			err := json.Unmarshal([]byte(tt.jsonData), &streamChunk)
			if err != nil {
				t.Fatalf("failed to unmarshal: %v", err)
			}

			// Convert to common format
			chunk := clients.StreamChunk{
				Done:     streamChunk.Type == "message_stop",
				Metadata: make(map[string]interface{}),
			}

			if streamChunk.Delta != nil && streamChunk.Delta.Type == "text_delta" {
				chunk.Delta = streamChunk.Delta.Text
			}

			if streamChunk.Delta != nil && streamChunk.Delta.StopReason != "" {
				chunk.Done = true
			}

			if chunk.Delta != tt.wantText {
				t.Errorf("got text %q, want %q", chunk.Delta, tt.wantText)
			}

			if chunk.Done != tt.wantDone {
				t.Errorf("got done %v, want %v", chunk.Done, tt.wantDone)
			}
		})
	}
}
