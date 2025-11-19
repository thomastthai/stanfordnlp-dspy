// Package bedrock provides an AWS Bedrock API client implementation.
package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/stanfordnlp/dspy/internal/clients"
)

const (
	defaultTimeout = 60 * time.Second
)

// Client is an AWS Bedrock API client.
type Client struct {
	client  *bedrockruntime.Client
	region  string
	timeout time.Duration
}

// ClientOptions configures the Bedrock client.
type ClientOptions struct {
	Region  string
	Timeout time.Duration
}

// NewClient creates a new Bedrock client.
func NewClient(opts ClientOptions) (*Client, error) {
	if opts.Region == "" {
		opts.Region = "us-east-1"
	}

	if opts.Timeout == 0 {
		opts.Timeout = defaultTimeout
	}

	// Load AWS configuration
	cfg, err := config.LoadDefaultConfig(context.Background(),
		config.WithRegion(opts.Region),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS config: %w", err)
	}

	// Create Bedrock Runtime client
	client := bedrockruntime.NewFromConfig(cfg)

	return &Client{
		client:  client,
		region:  opts.Region,
		timeout: opts.Timeout,
	}, nil
}

// Call sends a request to the Bedrock API.
func (c *Client) Call(ctx context.Context, request *clients.Request, modelID string) (*clients.Response, error) {
	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Determine model provider from model ID
	provider := getModelProvider(modelID)

	// Build request body based on provider
	requestBody, err := buildRequestBody(provider, request)
	if err != nil {
		return nil, fmt.Errorf("failed to build request body: %w", err)
	}

	// Invoke model
	output, err := c.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     &modelID,
		Body:        requestBody,
		ContentType: stringPtr("application/json"),
	})
	if err != nil {
		return nil, fmt.Errorf("Bedrock API call failed: %w", err)
	}

	// Parse response based on provider
	response, err := parseResponse(provider, output.Body, modelID)
	if err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return response, nil
}

// getModelProvider extracts the provider from the model ID.
func getModelProvider(modelID string) string {
	if strings.HasPrefix(modelID, "anthropic.") {
		return "anthropic"
	} else if strings.HasPrefix(modelID, "amazon.titan") {
		return "titan"
	} else if strings.HasPrefix(modelID, "meta.llama") {
		return "llama"
	} else if strings.HasPrefix(modelID, "ai21.") {
		return "ai21"
	} else if strings.HasPrefix(modelID, "cohere.") {
		return "cohere"
	}
	return "unknown"
}

// buildRequestBody builds the request body for the specific provider.
func buildRequestBody(provider string, request *clients.Request) ([]byte, error) {
	switch provider {
	case "anthropic":
		return buildAnthropicRequest(request)
	case "titan":
		return buildTitanRequest(request)
	case "llama":
		return buildLlamaRequest(request)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}
}

// buildAnthropicRequest builds a request for Anthropic Claude models.
func buildAnthropicRequest(request *clients.Request) ([]byte, error) {
	type Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	type Request struct {
		Messages      []Message `json:"messages"`
		MaxTokens     int       `json:"max_tokens"`
		Temperature   float64   `json:"temperature,omitempty"`
		TopP          float64   `json:"top_p,omitempty"`
		StopSequences []string  `json:"stop_sequences,omitempty"`
		AnthropicVersion string `json:"anthropic_version"`
	}

	messages := make([]Message, 0, len(request.Messages))
	for _, msg := range request.Messages {
		if msg.Role != "system" {
			messages = append(messages, Message{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
	}

	// If no messages, use prompt as user message
	if len(messages) == 0 && request.Prompt != "" {
		messages = append(messages, Message{
			Role:    "user",
			Content: request.Prompt,
		})
	}

	maxTokens := request.MaxTokens
	if maxTokens == 0 {
		maxTokens = 1000
	}

	req := Request{
		Messages:         messages,
		MaxTokens:        maxTokens,
		Temperature:      request.Temperature,
		TopP:             request.TopP,
		StopSequences:    request.StopSequences,
		AnthropicVersion: "bedrock-2023-05-31",
	}

	return json.Marshal(req)
}

// buildTitanRequest builds a request for Amazon Titan models.
func buildTitanRequest(request *clients.Request) ([]byte, error) {
	type TextGenerationConfig struct {
		Temperature   float64  `json:"temperature,omitempty"`
		TopP          float64  `json:"topP,omitempty"`
		MaxTokenCount int      `json:"maxTokenCount,omitempty"`
		StopSequences []string `json:"stopSequences,omitempty"`
	}

	type Request struct {
		InputText             string                `json:"inputText"`
		TextGenerationConfig  TextGenerationConfig  `json:"textGenerationConfig"`
	}

	// Build input text from messages or prompt
	inputText := request.Prompt
	if len(request.Messages) > 0 {
		for _, msg := range request.Messages {
			inputText += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
		}
	}

	maxTokens := request.MaxTokens
	if maxTokens == 0 {
		maxTokens = 1000
	}

	req := Request{
		InputText: inputText,
		TextGenerationConfig: TextGenerationConfig{
			Temperature:   request.Temperature,
			TopP:          request.TopP,
			MaxTokenCount: maxTokens,
			StopSequences: request.StopSequences,
		},
	}

	return json.Marshal(req)
}

// buildLlamaRequest builds a request for Meta Llama models.
func buildLlamaRequest(request *clients.Request) ([]byte, error) {
	type Request struct {
		Prompt      string   `json:"prompt"`
		Temperature float64  `json:"temperature,omitempty"`
		TopP        float64  `json:"top_p,omitempty"`
		MaxGenLen   int      `json:"max_gen_len,omitempty"`
	}

	// Build prompt from messages or use direct prompt
	prompt := request.Prompt
	if len(request.Messages) > 0 {
		for _, msg := range request.Messages {
			prompt += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
		}
	}

	maxTokens := request.MaxTokens
	if maxTokens == 0 {
		maxTokens = 1000
	}

	req := Request{
		Prompt:      prompt,
		Temperature: request.Temperature,
		TopP:        request.TopP,
		MaxGenLen:   maxTokens,
	}

	return json.Marshal(req)
}

// parseResponse parses the response based on provider.
func parseResponse(provider string, body []byte, modelID string) (*clients.Response, error) {
	switch provider {
	case "anthropic":
		return parseAnthropicResponse(body, modelID)
	case "titan":
		return parseTitanResponse(body, modelID)
	case "llama":
		return parseLlamaResponse(body, modelID)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}
}

// parseAnthropicResponse parses Anthropic Claude responses.
func parseAnthropicResponse(body []byte, modelID string) (*clients.Response, error) {
	type Content struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}

	type Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	}

	type Response struct {
		ID         string    `json:"id"`
		Content    []Content `json:"content"`
		StopReason string    `json:"stop_reason"`
		Usage      Usage     `json:"usage"`
	}

	var resp Response
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Extract text from content
	var text string
	for _, content := range resp.Content {
		if content.Type == "text" {
			text += content.Text
		}
	}

	return &clients.Response{
		ID:    resp.ID,
		Model: modelID,
		Choices: []clients.Choice{
			{
				Index:        0,
				FinishReason: resp.StopReason,
				Message: clients.Message{
					Role:    "assistant",
					Content: text,
				},
			},
		},
		Usage: clients.Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}, nil
}

// parseTitanResponse parses Amazon Titan responses.
func parseTitanResponse(body []byte, modelID string) (*clients.Response, error) {
	type Result struct {
		TokenCount       int    `json:"tokenCount"`
		OutputText       string `json:"outputText"`
		CompletionReason string `json:"completionReason"`
	}

	type Response struct {
		InputTextTokenCount int      `json:"inputTextTokenCount"`
		Results             []Result `json:"results"`
	}

	var resp Response
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(resp.Results) == 0 {
		return nil, fmt.Errorf("no results in response")
	}

	result := resp.Results[0]

	return &clients.Response{
		Model: modelID,
		Choices: []clients.Choice{
			{
				Index:        0,
				FinishReason: result.CompletionReason,
				Text:         result.OutputText,
			},
		},
		Usage: clients.Usage{
			PromptTokens:     resp.InputTextTokenCount,
			CompletionTokens: result.TokenCount,
			TotalTokens:      resp.InputTextTokenCount + result.TokenCount,
		},
	}, nil
}

// parseLlamaResponse parses Meta Llama responses.
func parseLlamaResponse(body []byte, modelID string) (*clients.Response, error) {
	type Response struct {
		Generation           string `json:"generation"`
		PromptTokenCount     int    `json:"prompt_token_count"`
		GenerationTokenCount int    `json:"generation_token_count"`
		StopReason           string `json:"stop_reason"`
	}

	var resp Response
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &clients.Response{
		Model: modelID,
		Choices: []clients.Choice{
			{
				Index:        0,
				FinishReason: resp.StopReason,
				Text:         resp.Generation,
			},
		},
		Usage: clients.Usage{
			PromptTokens:     resp.PromptTokenCount,
			CompletionTokens: resp.GenerationTokenCount,
			TotalTokens:      resp.PromptTokenCount + resp.GenerationTokenCount,
		},
	}, nil
}

// stringPtr returns a pointer to a string.
func stringPtr(s string) *string {
	return &s
}
