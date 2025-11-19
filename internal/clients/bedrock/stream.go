package bedrock

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stanfordnlp/dspy/internal/clients"
)

// Stream sends a streaming request to the Bedrock API.
func (c *Client) Stream(ctx context.Context, request *clients.Request, modelID string) (<-chan clients.StreamChunk, <-chan error) {
	chunkCh := make(chan clients.StreamChunk, 10)
	errCh := make(chan error, 1)

	go func() {
		defer close(chunkCh)
		defer close(errCh)

		// Determine model provider from model ID
		provider := getModelProvider(modelID)

		// Build request body based on provider
		requestBody, err := buildRequestBody(provider, request)
		if err != nil {
			errCh <- fmt.Errorf("failed to build request body: %w", err)
			return
		}

		// Invoke model with streaming
		output, err := c.client.InvokeModelWithResponseStream(ctx, &bedrockruntime.InvokeModelWithResponseStreamInput{
			ModelId:     &modelID,
			Body:        requestBody,
			ContentType: stringPtr("application/json"),
		})
		if err != nil {
			errCh <- fmt.Errorf("Bedrock API streaming call failed: %w", err)
			return
		}

		// Process the stream
		stream := output.GetStream()
		eventsCh := stream.Events()

		for {
			select {
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			case event, ok := <-eventsCh:
				if !ok {
					// Stream closed
					return
				}

				// Handle the stream event based on type
				switch e := event.(type) {
				case *types.ResponseStreamMemberChunk:
					if err := handleStreamChunk(e.Value.Bytes, provider, chunkCh); err != nil {
						errCh <- err
						return
					}
				}
			}
		}
	}()

	return chunkCh, errCh
}

// handleStreamChunk processes a single stream chunk.
func handleStreamChunk(data []byte, provider string, chunkCh chan<- clients.StreamChunk) error {
	// Parse the chunk based on provider
	switch provider {
	case "anthropic":
		return handleAnthropicStreamChunk(data, chunkCh)
	case "titan":
		return handleTitanStreamChunk(data, chunkCh)
	case "llama":
		return handleLlamaStreamChunk(data, chunkCh)
	default:
		return fmt.Errorf("streaming not supported for provider: %s", provider)
	}
}

// handleAnthropicStreamChunk handles Anthropic Claude streaming chunks.
func handleAnthropicStreamChunk(data []byte, chunkCh chan<- clients.StreamChunk) error {
	type StreamChunk struct {
		Type  string `json:"type"`
		Index int    `json:"index,omitempty"`
		Delta struct {
			Type string `json:"type"`
			Text string `json:"text,omitempty"`
		} `json:"delta,omitempty"`
		StopReason string `json:"stop_reason,omitempty"`
	}

	var chunk StreamChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil // Skip malformed chunks
	}

	// Only send content delta chunks
	if chunk.Type == "content_block_delta" && chunk.Delta.Type == "text_delta" {
		chunkCh <- clients.StreamChunk{
			Delta: chunk.Delta.Text,
			Done:  false,
			Metadata: map[string]interface{}{
				"type":  chunk.Type,
				"index": chunk.Index,
			},
		}
	} else if chunk.Type == "message_stop" || chunk.StopReason != "" {
		chunkCh <- clients.StreamChunk{
			Delta: "",
			Done:  true,
			Metadata: map[string]interface{}{
				"stop_reason": chunk.StopReason,
			},
		}
	}

	return nil
}

// handleTitanStreamChunk handles Amazon Titan streaming chunks.
func handleTitanStreamChunk(data []byte, chunkCh chan<- clients.StreamChunk) error {
	type StreamChunk struct {
		OutputText                string `json:"outputText"`
		Index                     int    `json:"index"`
		TotalOutputTextTokenCount int    `json:"totalOutputTextTokenCount,omitempty"`
		CompletionReason          string `json:"completionReason,omitempty"`
	}

	var chunk StreamChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil // Skip malformed chunks
	}

	done := chunk.CompletionReason != ""

	chunkCh <- clients.StreamChunk{
		Delta: chunk.OutputText,
		Done:  done,
		Metadata: map[string]interface{}{
			"index":             chunk.Index,
			"completion_reason": chunk.CompletionReason,
		},
	}

	return nil
}

// handleLlamaStreamChunk handles Meta Llama streaming chunks.
func handleLlamaStreamChunk(data []byte, chunkCh chan<- clients.StreamChunk) error {
	type StreamChunk struct {
		Generation string `json:"generation"`
		StopReason string `json:"stop_reason,omitempty"`
		TokenCount int    `json:"generation_token_count,omitempty"`
	}

	var chunk StreamChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil // Skip malformed chunks
	}

	done := chunk.StopReason != ""

	chunkCh <- clients.StreamChunk{
		Delta: chunk.Generation,
		Done:  done,
		Metadata: map[string]interface{}{
			"stop_reason": chunk.StopReason,
			"token_count": chunk.TokenCount,
		},
	}

	return nil
}
