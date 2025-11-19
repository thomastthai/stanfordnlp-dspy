package gemini

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

// Stream sends a streaming request to the Gemini API.
func (c *Client) Stream(ctx context.Context, request *clients.Request, model string) (<-chan clients.StreamChunk, <-chan error) {
	chunkCh := make(chan clients.StreamChunk, 10)
	errCh := make(chan error, 1)

	go func() {
		defer close(chunkCh)
		defer close(errCh)

		// Build contents
		contents := make([]Content, 0)

		for _, msg := range request.Messages {
			role := msg.Role
			// Map roles to Gemini's format
			if role == "assistant" {
				role = "model"
			}
			contents = append(contents, Content{
				Role: role,
				Parts: []Part{
					{Text: msg.Content},
				},
			})
		}

		// If no messages from request, use prompt
		if len(contents) == 0 && request.Prompt != "" {
			contents = append(contents, Content{
				Role: "user",
				Parts: []Part{
					{Text: request.Prompt},
				},
			})
		}

		// Build generation config
		genConfig := &GenerationConfig{}
		if request.Temperature > 0 {
			temp := request.Temperature
			genConfig.Temperature = &temp
		}
		if request.TopP > 0 {
			topP := request.TopP
			genConfig.TopP = &topP
		}
		if request.MaxTokens > 0 {
			genConfig.MaxOutputTokens = &request.MaxTokens
		}
		if len(request.StopSequences) > 0 {
			genConfig.StopSequences = request.StopSequences
		}
		if request.N > 0 {
			genConfig.CandidateCount = &request.N
		}

		// Build request body
		reqBody := GenerateContentRequest{
			Contents:         contents,
			GenerationConfig: genConfig,
		}

		// Marshal to JSON
		jsonData, err := json.Marshal(reqBody)
		if err != nil {
			errCh <- fmt.Errorf("failed to marshal request: %w", err)
			return
		}

		// Build URL for streaming endpoint
		url := fmt.Sprintf("%s/models/%s:streamGenerateContent?key=%s&alt=sse", c.baseURL, model, c.apiKey)

		// Create HTTP request
		httpReq, err := retryablehttp.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
		if err != nil {
			errCh <- fmt.Errorf("failed to create request: %w", err)
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")

		// Make the API call
		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			errCh <- fmt.Errorf("Gemini API call failed: %w", err)
			return
		}
		defer resp.Body.Close()

		// Check for errors
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			errCh <- fmt.Errorf("Gemini API error (status %d): %s", resp.StatusCode, string(body))
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

// parseSSEStream parses Server-Sent Events (SSE) stream from Gemini.
func (c *Client) parseSSEStream(ctx context.Context, reader io.Reader, chunkCh chan<- clients.StreamChunk) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024) // Increase buffer size for large responses

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
		if data == "[DONE]" || data == "" {
			continue
		}

		// Parse JSON chunk
		var geminiResp GenerateContentResponse
		if err := json.Unmarshal([]byte(data), &geminiResp); err != nil {
			// Skip malformed chunks
			continue
		}

		// Convert to common StreamChunk format
		if len(geminiResp.Candidates) > 0 {
			candidate := geminiResp.Candidates[0]

			// Extract text from parts
			var text string
			for _, part := range candidate.Content.Parts {
				text += part.Text
			}

			chunk := clients.StreamChunk{
				Delta: text,
				Done:  candidate.FinishReason != "",
				Metadata: map[string]interface{}{
					"finish_reason": candidate.FinishReason,
					"index":         candidate.Index,
				},
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
