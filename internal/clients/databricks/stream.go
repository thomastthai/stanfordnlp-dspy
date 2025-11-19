package databricks

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/hashicorp/go-retryablehttp"
	"github.com/stanfordnlp/dspy/internal/clients"
)

// StreamChunk represents a streaming response chunk from Databricks.
type StreamChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
}

// ChunkChoice represents a streaming choice.
type ChunkChoice struct {
	Index        int         `json:"index"`
	Delta        ChunkDelta  `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

// ChunkDelta represents the delta content in a streaming chunk.
type ChunkDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// Stream sends a streaming request to the Databricks Model Serving API.
func (c *Client) Stream(ctx context.Context, request *clients.Request, endpoint string) (<-chan clients.StreamChunk, <-chan error) {
	chunkCh := make(chan clients.StreamChunk, 10)
	errCh := make(chan error, 1)

	go func() {
		defer close(chunkCh)
		defer close(errCh)

		// Build request body
		servingReq := ServingRequest{
			Temperature: request.Temperature,
			MaxTokens:   request.MaxTokens,
			TopP:        request.TopP,
			Stop:        request.StopSequences,
			N:           request.N,
		}

		// Use messages if available, otherwise use prompt
		if len(request.Messages) > 0 {
			servingReq.Messages = make([]Message, len(request.Messages))
			for i, msg := range request.Messages {
				servingReq.Messages[i] = Message{
					Role:    msg.Role,
					Content: msg.Content,
				}
			}
		} else if request.Prompt != "" {
			servingReq.Prompt = request.Prompt
		}

		// Marshal to JSON
		jsonData, err := json.Marshal(servingReq)
		if err != nil {
			errCh <- fmt.Errorf("failed to marshal request: %w", err)
			return
		}

		// Build URL
		url := fmt.Sprintf("%s/serving-endpoints/%s/invocations", c.host, endpoint)

		// Create HTTP request
		httpReq, err := retryablehttp.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
		if err != nil {
			errCh <- fmt.Errorf("failed to create request: %w", err)
			return
		}

		// Set headers for streaming
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.token))
		httpReq.Header.Set("Accept", "text/event-stream")

		// Make the API call
		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			errCh <- fmt.Errorf("Databricks API call failed: %w", err)
			return
		}
		defer resp.Body.Close()

		// Check for errors
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			errCh <- fmt.Errorf("Databricks API error (status %d): %s", resp.StatusCode, string(body))
			return
		}

		// Parse SSE stream
		if err := parseDatabricksSSEStream(ctx, resp.Body, chunkCh); err != nil {
			errCh <- err
			return
		}
	}()

	return chunkCh, errCh
}

// parseDatabricksSSEStream parses the Databricks SSE stream.
func parseDatabricksSSEStream(ctx context.Context, reader io.Reader, chunkCh chan<- clients.StreamChunk) error {
	scanner := bufio.NewScanner(reader)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				return fmt.Errorf("error reading stream: %w", err)
			}
			// Stream ended normally
			return nil
		}

		line := scanner.Text()

		// SSE lines start with "data: "
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		// Check for stream end marker
		if data == "[DONE]" {
			return nil
		}

		// Parse the chunk
		var databricksChunk StreamChunk
		if err := json.Unmarshal([]byte(data), &databricksChunk); err != nil {
			// Skip malformed chunks
			continue
		}

		// Convert to standard StreamChunk
		for _, choice := range databricksChunk.Choices {
			chunk := clients.StreamChunk{
				Delta: choice.Delta.Content,
				Done:  choice.FinishReason != nil,
				Metadata: map[string]interface{}{
					"id":    databricksChunk.ID,
					"model": databricksChunk.Model,
					"index": choice.Index,
				},
			}

			select {
			case chunkCh <- chunk:
			case <-ctx.Done():
				return ctx.Err()
			}

			// If this is the final chunk, we're done
			if chunk.Done {
				return nil
			}
		}
	}
}
