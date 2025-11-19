// Package azure provides an Azure OpenAI API client implementation.
package azure

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
	defaultTimeout    = 60 * time.Second
	defaultAPIVersion = "2024-02-15-preview"
)

// Client is an Azure OpenAI API client.
type Client struct {
	apiKey     string
	endpoint   string
	apiVersion string
	httpClient *retryablehttp.Client
	timeout    time.Duration
}

// ClientOptions configures the Azure OpenAI client.
type ClientOptions struct {
	Endpoint   string
	APIKey     string
	APIVersion string
	Timeout    time.Duration
}

// NewClient creates a new Azure OpenAI client.
func NewClient(opts ClientOptions) (*Client, error) {
	if opts.Endpoint == "" {
		return nil, fmt.Errorf("Azure OpenAI endpoint is required")
	}

	if opts.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	if opts.APIVersion == "" {
		opts.APIVersion = defaultAPIVersion
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
		endpoint:   opts.Endpoint,
		apiVersion: opts.APIVersion,
		httpClient: retryClient,
		timeout:    opts.Timeout,
	}, nil
}

// AzureMessage represents a message in the Azure OpenAI format.
type AzureMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// AzureRequest represents an Azure OpenAI API request.
type AzureRequest struct {
	Messages    []AzureMessage `json:"messages"`
	Temperature float64        `json:"temperature,omitempty"`
	MaxTokens   int            `json:"max_tokens,omitempty"`
	TopP        float64        `json:"top_p,omitempty"`
	Stop        []string       `json:"stop,omitempty"`
	N           int            `json:"n,omitempty"`
}

// AzureResponse represents an Azure OpenAI API response.
type AzureResponse struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []AzureChoice `json:"choices"`
	Usage   AzureUsage    `json:"usage"`
}

// AzureChoice represents a completion choice.
type AzureChoice struct {
	Index        int          `json:"index"`
	Message      AzureMessage `json:"message"`
	FinishReason string       `json:"finish_reason"`
}

// AzureUsage represents token usage information.
type AzureUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Call sends a request to the Azure OpenAI API.
func (c *Client) Call(ctx context.Context, request *clients.Request, deploymentName string) (*clients.Response, error) {
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
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Build URL
	url := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s",
		c.endpoint, deploymentName, c.apiVersion)

	// Create HTTP request
	httpReq, err := retryablehttp.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("api-key", c.apiKey)

	// Make the API call
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Azure OpenAI API call failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Azure OpenAI API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var azureResp AzureResponse
	if err := json.Unmarshal(body, &azureResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Convert to our response format
	response := &clients.Response{
		ID:      azureResp.ID,
		Model:   azureResp.Model,
		Choices: make([]clients.Choice, 0, len(azureResp.Choices)),
		Usage: clients.Usage{
			PromptTokens:     azureResp.Usage.PromptTokens,
			CompletionTokens: azureResp.Usage.CompletionTokens,
			TotalTokens:      azureResp.Usage.TotalTokens,
		},
	}

	for _, choice := range azureResp.Choices {
		response.Choices = append(response.Choices, clients.Choice{
			Index:        choice.Index,
			FinishReason: choice.FinishReason,
			Message: clients.Message{
				Role:    choice.Message.Role,
				Content: choice.Message.Content,
			},
		})
	}

	return response, nil
}
