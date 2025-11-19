package anthropic

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

// StreamChunk represents a streaming response chunk from Anthropic.
type StreamChunk struct {
	Type  string          `json:"type"`
	Index int             `json:"index,omitempty"`
	Delta *StreamDelta    `json:"delta,omitempty"`
	Message *AnthropicResponse `json:"message,omitempty"`
}

// StreamDelta represents the incremental content in a streaming chunk.
type StreamDelta struct {
	Type         string `json:"type"`
	Text         string `json:"text,omitempty"`
	StopReason   string `json:"stop_reason,omitempty"`
	StopSequence string `json:"stop_sequence,omitempty"`
}

// Stream sends a streaming request to the Anthropic API.
func (c *Client) Stream(ctx context.Context, request *clients.Request, model string) (<-chan clients.StreamChunk, <-chan error) {
	chunkCh := make(chan clients.StreamChunk, 10)
	errCh := make(chan error, 1)

	go func() {
		defer close(chunkCh)
		defer close(errCh)

		// Build messages
		messages := make([]AnthropicMessage, 0, len(request.Messages))
		var systemPrompt string

		for _, msg := range request.Messages {
			if msg.Role == "system" {
				systemPrompt = msg.Content
				continue
			}
			messages = append(messages, AnthropicMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}

		// If no messages from request, use prompt
		if len(messages) == 0 && request.Prompt != "" {
			messages = append(messages, AnthropicMessage{
				Role:    "user",
				Content: request.Prompt,
			})
		}

		// Build request params
		maxTokens := request.MaxTokens
		if maxTokens == 0 {
			maxTokens = defaultMaxTokens
		}

		reqBody := AnthropicRequest{
			Model:         model,
			Messages:      messages,
			MaxTokens:     maxTokens,
			Temperature:   request.Temperature,
			TopP:          request.TopP,
			StopSequences: request.StopSequences,
			System:        systemPrompt,
			Stream:        true,
		}

		// Marshal to JSON
		jsonData, err := json.Marshal(reqBody)
		if err != nil {
			errCh <- fmt.Errorf("failed to marshal request: %w", err)
			return
		}

		// Create HTTP request
		url := fmt.Sprintf("%s/v1/messages", c.baseURL)
		httpReq, err := retryablehttp.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
		if err != nil {
			errCh <- fmt.Errorf("failed to create request: %w", err)
			return
		}

		// Set headers
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("x-api-key", c.apiKey)
		httpReq.Header.Set("anthropic-version", apiVersion)

		// Make the API call
		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			errCh <- fmt.Errorf("anthropic API call failed: %w", err)
			return
		}
		defer resp.Body.Close()

		// Check for errors
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			errCh <- fmt.Errorf("Anthropic API error (status %d): %s", resp.StatusCode, string(body))
			return
		}

		// Parse SSE stream
		if err := c.parseSSEStream(ctx, resp.Body, chunkCh); err != nil {
			errCh <- err
			return
		}
	}()

	return chunkCh, errCh
}

// parseSSEStream parses Server-Sent Events (SSE) stream from Anthropic.
func (c *Client) parseSSEStream(ctx context.Context, reader io.Reader, chunkCh chan<- clients.StreamChunk) error {
	scanner := bufio.NewScanner(reader)

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line := scanner.Text()

		// SSE lines start with "data: " or "event: "
		if strings.HasPrefix(line, "event: ") {
			// Event type (message_start, content_block_delta, message_stop, etc.)
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		// Check for stream end marker
		if data == "[DONE]" {
			return nil
		}

		// Parse JSON chunk
		var streamChunk StreamChunk
		if err := json.Unmarshal([]byte(data), &streamChunk); err != nil {
			// Skip malformed chunks
			continue
		}

		// Convert to common StreamChunk format
		chunk := clients.StreamChunk{
			Done:     streamChunk.Type == "message_stop",
			Metadata: make(map[string]interface{}),
		}

		// Extract text delta
		if streamChunk.Delta != nil && streamChunk.Delta.Type == "text_delta" {
			chunk.Delta = streamChunk.Delta.Text
		}

		// Handle completion
		if streamChunk.Delta != nil && streamChunk.Delta.StopReason != "" {
			chunk.Done = true
			chunk.Metadata["stop_reason"] = streamChunk.Delta.StopReason
		}

		// Send chunk
		select {
		case chunkCh <- chunk:
		case <-ctx.Done():
			return ctx.Err()
		}

		if chunk.Done {
			return nil
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading stream: %w", err)
	}

	return nil
}

// StreamToText collects all chunks and returns the complete text.
func StreamToText(chunkCh <-chan clients.StreamChunk, errCh <-chan error) (string, error) {
	return clients.CollectStreamText(chunkCh, errCh)
}
