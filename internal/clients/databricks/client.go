// Package databricks provides a Databricks Model Serving API client implementation.
package databricks

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
	defaultTimeout = 60 * time.Second
)

// Client is a Databricks Model Serving API client.
type Client struct {
	host       string
	token      string
	httpClient *retryablehttp.Client
}

// ClientOptions configures the Databricks client.
type ClientOptions struct {
	Host    string
	Token   string
	Timeout time.Duration
}

// NewClient creates a new Databricks client.
func NewClient(opts ClientOptions) (*Client, error) {
	if opts.Host == "" {
		return nil, fmt.Errorf("Databricks host is required")
	}

	if opts.Token == "" {
		return nil, fmt.Errorf("Databricks token is required")
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
		host:       opts.Host,
		token:      opts.Token,
		httpClient: retryClient,
	}, nil
}

// ServingRequest represents a Databricks model serving request.
type ServingRequest struct {
	Messages    []Message `json:"messages,omitempty"`
	Prompt      string    `json:"prompt,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	TopP        float64   `json:"top_p,omitempty"`
	Stop        []string  `json:"stop,omitempty"`
	N           int       `json:"n,omitempty"`
}

// Message represents a chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ServingResponse represents a Databricks model serving response.
type ServingResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Choice represents a completion choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	Text         string  `json:"text"`
	FinishReason string  `json:"finish_reason"`
}

// Usage represents token usage information.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Call sends a request to the Databricks Model Serving API.
func (c *Client) Call(ctx context.Context, request *clients.Request, endpoint string) (*clients.Response, error) {
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
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Build URL
	url := fmt.Sprintf("%s/serving-endpoints/%s/invocations", c.host, endpoint)

	// Create HTTP request
	httpReq, err := retryablehttp.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.token))

	// Make the API call
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Databricks API call failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Databricks API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var servingResp ServingResponse
	if err := json.Unmarshal(body, &servingResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Convert to our response format
	response := &clients.Response{
		ID:      servingResp.ID,
		Model:   servingResp.Model,
		Choices: make([]clients.Choice, 0, len(servingResp.Choices)),
		Usage: clients.Usage{
			PromptTokens:     servingResp.Usage.PromptTokens,
			CompletionTokens: servingResp.Usage.CompletionTokens,
			TotalTokens:      servingResp.Usage.TotalTokens,
		},
	}

	for _, choice := range servingResp.Choices {
		clientChoice := clients.Choice{
			Index:        choice.Index,
			FinishReason: choice.FinishReason,
		}

		// Handle both message and text formats
		if choice.Message.Content != "" {
			clientChoice.Message = clients.Message{
				Role:    choice.Message.Role,
				Content: choice.Message.Content,
			}
		} else {
			clientChoice.Text = choice.Text
		}

		response.Choices = append(response.Choices, clientChoice)
	}

	return response, nil
}
