package azure

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stanfordnlp/dspy/internal/clients"
)

func TestClient_Stream(t *testing.T) {
	// Create a mock SSE server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify headers
		if r.Header.Get("api-key") == "" {
			t.Error("expected api-key header")
		}
		if r.Header.Get("Accept") != "text/event-stream" {
			t.Error("expected Accept: text/event-stream header")
		}

		// Send SSE stream
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Send chunks
		chunks := []string{
			`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
			`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" World"},"finish_reason":null}]}`,
			`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			`data: [DONE]`,
		}

		for _, chunk := range chunks {
			w.Write([]byte(chunk + "\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	defer server.Close()

	// Create client
	client, err := NewClient(ClientOptions{
		Endpoint:   server.URL,
		APIKey:     "test-key",
		APIVersion: "2024-02-15-preview",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	// Create request
	request := &clients.Request{
		Messages: []clients.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	// Stream
	ctx := context.Background()
	chunkCh, errCh := client.Stream(ctx, request, "gpt-4")

	// Collect chunks
	var chunks []string
	for {
		select {
		case chunk, ok := <-chunkCh:
			if !ok {
				// Check for errors
				select {
				case err := <-errCh:
					if err != nil {
						t.Fatalf("stream error: %v", err)
					}
				default:
				}
				goto done
			}
			chunks = append(chunks, chunk.Delta)
		case err := <-errCh:
			if err != nil {
				t.Fatalf("stream error: %v", err)
			}
		}
	}

done:
	// Verify chunks
	if len(chunks) < 2 {
		t.Errorf("expected at least 2 chunks, got %d", len(chunks))
	}

	// Verify content
	fullText := ""
	for _, chunk := range chunks {
		fullText += chunk
	}
	if fullText != "Hello World" {
		t.Errorf("expected 'Hello World', got '%s'", fullText)
	}
}

func TestClient_Stream_Error(t *testing.T) {
	// Create a mock server that returns an error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error":{"code":"invalid_api_key","message":"Invalid API key"}}`))
	}))
	defer server.Close()

	// Create client
	client, err := NewClient(ClientOptions{
		Endpoint:   server.URL,
		APIKey:     "invalid-key",
		APIVersion: "2024-02-15-preview",
	})
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	// Create request
	request := &clients.Request{
		Messages: []clients.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	// Stream
	ctx := context.Background()
	chunkCh, errCh := client.Stream(ctx, request, "gpt-4")

	// Should get an error
	select {
	case _, ok := <-chunkCh:
		if ok {
			t.Error("expected error, got chunk")
		}
	case err := <-errCh:
		if err == nil {
			t.Error("expected error, got nil")
		}
	}
}
