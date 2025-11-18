package openai

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
)

// StreamChunk represents a streaming response chunk.
type StreamChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
}

// ChunkChoice represents a streaming choice.
type ChunkChoice struct {
	Index        int              `json:"index"`
	Delta        ChunkDelta       `json:"delta"`
	FinishReason *string          `json:"finish_reason"`
	Logprobs     interface{}      `json:"logprobs,omitempty"`
}

// ChunkDelta represents the delta content in a streaming chunk.
type ChunkDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ChatCompletionStream sends a streaming chat completion request.
func (c *Client) ChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (<-chan StreamChunk, <-chan error) {
	chunkCh := make(chan StreamChunk, 10)
	errCh := make(chan error, 1)

	go func() {
		defer close(chunkCh)
		defer close(errCh)

		// Enable streaming
		req.Stream = true

		reqBody, err := json.Marshal(req)
		if err != nil {
			errCh <- fmt.Errorf("failed to marshal request: %w", err)
			return
		}

		httpReq, err := retryablehttp.NewRequestWithContext(ctx, "POST", c.baseURL+"/chat/completions", bytes.NewReader(reqBody))
		if err != nil {
			errCh <- fmt.Errorf("failed to create request: %w", err)
			return
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
		httpReq.Header.Set("Accept", "text/event-stream")
		if c.orgID != "" {
			httpReq.Header.Set("OpenAI-Organization", c.orgID)
		}

		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			errCh <- fmt.Errorf("failed to send request: %w", err)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			var errResp ErrorResponse
			if err := json.Unmarshal(body, &errResp); err != nil {
				errCh <- fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
				return
			}
			errCh <- fmt.Errorf("API error: %s (type: %s)", errResp.Error.Message, errResp.Error.Type)
			return
		}

		// Parse SSE stream
		if err := parseSSEStream(ctx, resp.Body, chunkCh); err != nil {
			errCh <- err
			return
		}
	}()

	return chunkCh, errCh
}

// parseSSEStream parses Server-Sent Events (SSE) stream.
func parseSSEStream(ctx context.Context, reader io.Reader, chunkCh chan<- StreamChunk) error {
	scanner := bufio.NewScanner(reader)
	
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
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

		// Parse JSON chunk
		var chunk StreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			// Skip malformed chunks
			continue
		}

		// Send chunk
		select {
		case chunkCh <- chunk:
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading stream: %w", err)
	}

	return nil
}

// StreamToText collects all chunks and returns the complete text.
func StreamToText(chunkCh <-chan StreamChunk, errCh <-chan error) (string, error) {
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

			// Append content from the chunk
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				builder.WriteString(chunk.Choices[0].Delta.Content)
			}

		case err := <-errCh:
			if err != nil {
				return "", err
			}
		}
	}
}
