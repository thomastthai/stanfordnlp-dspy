package databricks

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
		authHeader := r.Header.Get("Authorization")
		if authHeader != "Bearer test-token" {
			t.Errorf("expected Authorization header 'Bearer test-token', got '%s'", authHeader)
		}
		if r.Header.Get("Accept") != "text/event-stream" {
			t.Error("expected Accept: text/event-stream header")
		}

		// Send SSE stream
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Send chunks
		chunks := []string{
			`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"dbrx","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
			`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"dbrx","choices":[{"index":0,"delta":{"content":" from"},"finish_reason":null}]}`,
			`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"dbrx","choices":[{"index":0,"delta":{"content":" Databricks"},"finish_reason":null}]}`,
			`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"dbrx","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
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
		Host:  server.URL,
		Token: "test-token",
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
	chunkCh, errCh := client.Stream(ctx, request, "dbrx-instruct")

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
	if fullText != "Hello from Databricks" {
		t.Errorf("expected 'Hello from Databricks', got '%s'", fullText)
	}
}

func TestClient_Stream_Error(t *testing.T) {
	// Create a mock server that returns an error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error_code":"UNAUTHENTICATED","message":"Invalid token"}`))
	}))
	defer server.Close()

	// Create client
	client, err := NewClient(ClientOptions{
		Host:  server.URL,
		Token: "invalid-token",
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
	chunkCh, errCh := client.Stream(ctx, request, "dbrx-instruct")

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
