// Package openai provides an OpenAI API client implementation.
package openai

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
	defaultBaseURL = "https://api.openai.com/v1"
	defaultTimeout = 60 * time.Second
)

// Client is an OpenAI API client.
type Client struct {
	apiKey     string
	baseURL    string
	httpClient *retryablehttp.Client
	orgID      string
}

// ClientOptions configures the OpenAI client.
type ClientOptions struct {
	APIKey  string
	BaseURL string
	OrgID   string
	Timeout time.Duration
}

// NewClient creates a new OpenAI client.
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
	retryClient.CheckRetry = retryPolicy
	retryClient.HTTPClient.Timeout = opts.Timeout

	return &Client{
		apiKey:     opts.APIKey,
		baseURL:    opts.BaseURL,
		httpClient: retryClient,
		orgID:      opts.OrgID,
	}, nil
}

// retryPolicy determines whether to retry a request based on the response.
func retryPolicy(ctx context.Context, resp *http.Response, err error) (bool, error) {
	// Always retry on network errors
	if err != nil {
		return true, err
	}

	// Retry on rate limit (429) and server errors (5xx)
	if resp.StatusCode == 429 || resp.StatusCode >= 500 {
		return true, nil
	}

	// Don't retry on client errors (4xx except 429)
	if resp.StatusCode >= 400 && resp.StatusCode < 500 {
		return false, nil
	}

	return false, nil
}

// ChatCompletionRequest represents an OpenAI chat completion request.
type ChatCompletionRequest struct {
	Model            string                   `json:"model"`
	Messages         []ChatMessage            `json:"messages"`
	Temperature      *float64                 `json:"temperature,omitempty"`
	MaxTokens        *int                     `json:"max_tokens,omitempty"`
	TopP             *float64                 `json:"top_p,omitempty"`
	N                *int                     `json:"n,omitempty"`
	Stream           bool                     `json:"stream,omitempty"`
	Stop             []string                 `json:"stop,omitempty"`
	PresencePenalty  *float64                 `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64                 `json:"frequency_penalty,omitempty"`
	LogitBias        map[string]float64       `json:"logit_bias,omitempty"`
	User             string                   `json:"user,omitempty"`
	Functions        []FunctionDefinition     `json:"functions,omitempty"`
	FunctionCall     interface{}              `json:"function_call,omitempty"`
	Tools            []Tool                   `json:"tools,omitempty"`
	ToolChoice       interface{}              `json:"tool_choice,omitempty"`
	ResponseFormat   *ResponseFormat          `json:"response_format,omitempty"`
}

// ChatMessage represents a chat message.
type ChatMessage struct {
	Role       string      `json:"role"`
	Content    string      `json:"content"`
	Name       string      `json:"name,omitempty"`
	ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
}

// ToolCall represents a tool call.
type ToolCall struct {
	ID       string              `json:"id"`
	Type     string              `json:"type"`
	Function FunctionCallDetails `json:"function"`
}

// FunctionCallDetails represents function call details.
type FunctionCallDetails struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// FunctionDefinition represents a function definition.
type FunctionDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// Tool represents a tool definition.
type Tool struct {
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`
}

// ResponseFormat specifies the output format.
type ResponseFormat struct {
	Type string `json:"type"` // "text" or "json_object"
}

// ChatCompletionResponse represents an OpenAI chat completion response.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Choice represents a completion choice.
type Choice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
	Logprobs     interface{} `json:"logprobs,omitempty"`
}

// Usage represents token usage information.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ErrorResponse represents an OpenAI error response.
type ErrorResponse struct {
	Error ErrorDetails `json:"error"`
}

// ErrorDetails contains error details.
type ErrorDetails struct {
	Message string      `json:"message"`
	Type    string      `json:"type"`
	Param   string      `json:"param,omitempty"`
	Code    interface{} `json:"code,omitempty"`
}

// ChatCompletion sends a chat completion request to the OpenAI API.
func (c *Client) ChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := retryablehttp.NewRequestWithContext(ctx, "POST", c.baseURL+"/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	if c.orgID != "" {
		httpReq.Header.Set("OpenAI-Organization", c.orgID)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp ErrorResponse
		if err := json.Unmarshal(body, &errResp); err != nil {
			return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
		}
		return nil, fmt.Errorf("API error: %s (type: %s, code: %v)", errResp.Error.Message, errResp.Error.Type, errResp.Error.Code)
	}

	var completionResp ChatCompletionResponse
	if err := json.Unmarshal(body, &completionResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &completionResp, nil
}

// Call implements the BaseLM interface by converting Request to OpenAI format.
func (c *Client) Call(ctx context.Context, request *clients.Request, modelName string) (*clients.Response, error) {
	// Convert DSPy request to OpenAI format
	chatReq := ChatCompletionRequest{
		Model:    modelName,
		Messages: make([]ChatMessage, len(request.Messages)),
		Stream:   false,
	}

	// Convert messages
	for i, msg := range request.Messages {
		chatReq.Messages[i] = ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
			Name:    msg.Name,
		}

		// Convert tool calls if present
		if len(msg.ToolCalls) > 0 {
			chatReq.Messages[i].ToolCalls = make([]ToolCall, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				chatReq.Messages[i].ToolCalls[j] = ToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: FunctionCallDetails{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}
	}

	// Set optional parameters
	if request.Temperature > 0 || request.Temperature == 0 {
		temp := request.Temperature
		chatReq.Temperature = &temp
	}
	if request.MaxTokens > 0 {
		chatReq.MaxTokens = &request.MaxTokens
	}
	if request.TopP > 0 {
		chatReq.TopP = &request.TopP
	}
	if request.N > 0 {
		chatReq.N = &request.N
	}
	if len(request.StopSequences) > 0 {
		chatReq.Stop = request.StopSequences
	}

	// Make API call
	resp, err := c.ChatCompletion(ctx, chatReq)
	if err != nil {
		return nil, err
	}

	// Convert response back to DSPy format
	dspyResp := &clients.Response{
		Choices: make([]clients.Choice, len(resp.Choices)),
		Usage: clients.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
		Model:    resp.Model,
		ID:       resp.ID,
		Metadata: make(map[string]interface{}),
	}

	for i, choice := range resp.Choices {
		dspyMsg := clients.Message{
			Role:    choice.Message.Role,
			Content: choice.Message.Content,
			Name:    choice.Message.Name,
		}

		// Convert tool calls back
		if len(choice.Message.ToolCalls) > 0 {
			dspyMsg.ToolCalls = make([]clients.ToolCall, len(choice.Message.ToolCalls))
			for j, tc := range choice.Message.ToolCalls {
				dspyMsg.ToolCalls[j] = clients.ToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: clients.FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		dspyResp.Choices[i] = clients.Choice{
			Message:      dspyMsg,
			Text:         choice.Message.Content,
			Index:        choice.Index,
			FinishReason: choice.FinishReason,
			Logprobs:     choice.Logprobs,
		}
	}

	return dspyResp, nil
}
