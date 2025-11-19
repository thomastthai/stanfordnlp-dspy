// Package anthropic provides an Anthropic API client implementation.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/hashicorp/go-retryablehttp"
	"github.com/stanfordnlp/dspy/internal/clients"
)

const (
	defaultBaseURL   = "https://api.anthropic.com"
	defaultMaxTokens = 4096
	defaultTimeout   = 60 * time.Second
	apiVersion       = "2023-06-01"
)

// Client is an Anthropic API client.
type Client struct {
	apiKey     string
	baseURL    string
	httpClient *retryablehttp.Client
	timeout    time.Duration
}

// ClientOptions configures the Anthropic client.
type ClientOptions struct {
	APIKey  string
	BaseURL string
	Timeout time.Duration
}

// NewClient creates a new Anthropic client.
func NewClient(opts ClientOptions) (*Client, error) {
	if opts.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	if opts.BaseURL == "" {
		opts.BaseURL = defaultBaseURL
	}

	if opts.Timeout == 0 {
		opts.Timeout = defaultTimeout
	}

	// Create retryable HTTP client
	retryClient := retryablehttp.NewClient()
	retryClient.RetryMax = 3
	retryClient.RetryWaitMin = 1 * time.Second
	retryClient.RetryWaitMax = 10 * time.Second
	retryClient.HTTPClient.Timeout = opts.Timeout

	return &Client{
		apiKey:     opts.APIKey,
		baseURL:    opts.BaseURL,
		httpClient: retryClient,
		timeout:    opts.Timeout,
	}, nil
}

// AnthropicMessage represents a message in the Anthropic API format.
type AnthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// AnthropicRequest represents an Anthropic API request.
type AnthropicRequest struct {
	Model         string             `json:"model"`
	Messages      []AnthropicMessage `json:"messages"`
	MaxTokens     int                `json:"max_tokens"`
	Temperature   float64            `json:"temperature,omitempty"`
	TopP          float64            `json:"top_p,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	System        string             `json:"system,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	Tools         []Tool             `json:"tools,omitempty"`
}

// AnthropicResponse represents an Anthropic API response.
type AnthropicResponse struct {
	ID         string                   `json:"id"`
	Type       string                   `json:"type"`
	Role       string                   `json:"role"`
	Content    []AnthropicContentBlock  `json:"content"`
	Model      string                   `json:"model"`
	StopReason string                   `json:"stop_reason"`
	Usage      AnthropicUsage           `json:"usage"`
}

// AnthropicContentBlock represents a content block in the response.
type AnthropicContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// AnthropicUsage represents token usage information.
type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Call sends a request to the Anthropic API.
func (c *Client) Call(ctx context.Context, request *clients.Request, model string) (*clients.Response, error) {
	// Build messages
	messages := make([]AnthropicMessage, 0, len(request.Messages))
	var systemPrompt string

	for _, msg := range request.Messages {
		if msg.Role == "system" {
			// Anthropic handles system messages separately
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
	}

	// Marshal to JSON
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	url := fmt.Sprintf("%s/v1/messages", c.baseURL)
	httpReq, err := retryablehttp.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", c.apiKey)
	httpReq.Header.Set("anthropic-version", apiVersion)

	// Make the API call
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic API call failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Anthropic API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var anthropicResp AnthropicResponse
	if err := json.Unmarshal(body, &anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Extract text content
	var text string
	for _, block := range anthropicResp.Content {
		if block.Type == "text" {
			text += block.Text
		}
	}

	// Convert response to our format
	response := &clients.Response{
		ID:    anthropicResp.ID,
		Model: anthropicResp.Model,
		Choices: []clients.Choice{
			{
				Index:        0,
				FinishReason: anthropicResp.StopReason,
				Message: clients.Message{
					Role:    "assistant",
					Content: text,
				},
			},
		},
		Usage: clients.Usage{
			PromptTokens:     anthropicResp.Usage.InputTokens,
			CompletionTokens: anthropicResp.Usage.OutputTokens,
			TotalTokens:      anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens,
		},
	}

	return response, nil
}

// CountTokens estimates token count for Anthropic models.
// This is a rough approximation; for accurate counting, use the official API.
func CountTokens(text string) int {
	// Rough approximation: ~4 characters per token for English text
	return len(text) / 4
}

// GetModelInfo returns information about the model.
type ModelInfo struct {
	Name           string
	ContextWindow  int
	MaxTokens      int
	SupportsVision bool
	CostPer1MInput float64
	CostPer1MOutput float64
}

// GetModelInfo returns model-specific information.
func GetModelInfo(model string) ModelInfo {
	switch model {
	case "claude-3-opus-20240229", "claude-3-opus":
		return ModelInfo{
			Name:            model,
			ContextWindow:   200000,
			MaxTokens:       4096,
			SupportsVision:  true,
			CostPer1MInput:  15.00,
			CostPer1MOutput: 75.00,
		}
	case "claude-3-5-sonnet-20240620", "claude-3-5-sonnet":
		return ModelInfo{
			Name:            model,
			ContextWindow:   200000,
			MaxTokens:       4096,
			SupportsVision:  true,
			CostPer1MInput:  3.00,
			CostPer1MOutput: 15.00,
		}
	case "claude-3-sonnet-20240229", "claude-3-sonnet":
		return ModelInfo{
			Name:            model,
			ContextWindow:   200000,
			MaxTokens:       4096,
			SupportsVision:  true,
			CostPer1MInput:  3.00,
			CostPer1MOutput: 15.00,
		}
	case "claude-3-haiku-20240307", "claude-3-haiku":
		return ModelInfo{
			Name:            model,
			ContextWindow:   200000,
			MaxTokens:       4096,
			SupportsVision:  true,
			CostPer1MInput:  0.25,
			CostPer1MOutput: 1.25,
		}
	default:
		return ModelInfo{
			Name:          model,
			ContextWindow: 100000,
			MaxTokens:     4096,
		}
	}
}
