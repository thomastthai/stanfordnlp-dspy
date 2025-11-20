package azure

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/hashicorp/go-retryablehttp"
	"github.com/stanfordnlp/dspy/internal/clients"
)

// StreamChunk represents a streaming response chunk from Azure OpenAI.
type StreamChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
}

// ChunkChoice represents a streaming choice.
type ChunkChoice struct {
	Index        int        `json:"index"`
	Delta        ChunkDelta `json:"delta"`
	FinishReason *string    `json:"finish_reason"`
}

// ChunkDelta represents the delta content in a streaming chunk.
type ChunkDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// Stream sends a streaming request to the Azure OpenAI API.
func (c *Client) Stream(ctx context.Context, request *clients.Request, deploymentName string) (<-chan clients.StreamChunk, <-chan error) {
	chunkCh := make(chan clients.StreamChunk, 10)
	errCh := make(chan error, 1)

	go func() {
		defer close(chunkCh)
		defer close(errCh)

		// Build messages
		messages := make([]AzureMessage, 0, len(request.Messages))

		for _, msg := range request.Messages {
			messages = append(messages, AzureMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}

		// If no messages from request, use prompt as user message
		if len(messages) == 0 && request.Prompt != "" {
			messages = append(messages, AzureMessage{
				Role:    "user",
				Content: request.Prompt,
			})
		}

		// Build request body
		maxTokens := request.MaxTokens
		if maxTokens == 0 {
			maxTokens = 1000
		}

		reqBody := AzureRequest{
			Messages:    messages,
			Temperature: request.Temperature,
			MaxTokens:   maxTokens,
			TopP:        request.TopP,
			Stop:        request.StopSequences,
			N:           request.N,
		}

		// Marshal to JSON
		jsonData, err := json.Marshal(reqBody)
		if err != nil {
			errCh <- fmt.Errorf("failed to marshal request: %w", err)
			return
		}

		// Build URL
		url := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s",
			c.endpoint, deploymentName, c.apiVersion)

		// Create HTTP request
		httpReq, err := retryablehttp.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
		if err != nil {
			errCh <- fmt.Errorf("failed to create request: %w", err)
			return
		}

		// Set headers for streaming
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("api-key", c.apiKey)
		httpReq.Header.Set("Accept", "text/event-stream")

		// Make the API call
		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			errCh <- fmt.Errorf("Azure OpenAI API call failed: %w", err)
			return
		}
		defer resp.Body.Close()

		// Check for errors
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			errCh <- fmt.Errorf("Azure OpenAI API error (status %d): %s", resp.StatusCode, string(body))
			return
		}

		// Parse SSE stream using shared utility
		if err := parseAzureSSEStream(ctx, resp.Body, chunkCh); err != nil {
			errCh <- err
			return
		}
	}()

	return chunkCh, errCh
}

// parseAzureSSEStream parses the Azure OpenAI SSE stream.
func parseAzureSSEStream(ctx context.Context, reader io.Reader, chunkCh chan<- clients.StreamChunk) error {
	sseReader := clients.NewSSEReader(reader)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		data, err := sseReader.ReadEvent()
		if err != nil {
			if err == io.EOF {
				// Stream ended normally
				return nil
			}
			return fmt.Errorf("error reading event: %w", err)
		}

		// Parse the chunk
		var azureChunk StreamChunk
		if err := json.Unmarshal(data, &azureChunk); err != nil {
			// Skip malformed chunks
			continue
		}

		// Convert to standard StreamChunk
		for _, choice := range azureChunk.Choices {
			chunk := clients.StreamChunk{
				Delta: choice.Delta.Content,
				Done:  choice.FinishReason != nil,
				Metadata: map[string]interface{}{
					"id":    azureChunk.ID,
					"model": azureChunk.Model,
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
