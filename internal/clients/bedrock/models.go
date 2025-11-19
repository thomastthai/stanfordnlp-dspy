package bedrock

import (
	"encoding/json"
	"fmt"

	"github.com/stanfordnlp/dspy/internal/clients"
)

// ModelAdapter provides model-specific request/response formatting.
type ModelAdapter interface {
	BuildRequest(request *clients.Request) ([]byte, error)
	ParseResponse(body []byte, modelID string) (*clients.Response, error)
}

// AnthropicAdapter handles Anthropic Claude models.
type AnthropicAdapter struct{}

// BuildRequest builds an Anthropic-formatted request.
func (a *AnthropicAdapter) BuildRequest(request *clients.Request) ([]byte, error) {
	type Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	type Request struct {
		Messages         []Message `json:"messages"`
		MaxTokens        int       `json:"max_tokens"`
		Temperature      float64   `json:"temperature,omitempty"`
		TopP             float64   `json:"top_p,omitempty"`
		TopK             int       `json:"top_k,omitempty"`
		StopSequences    []string  `json:"stop_sequences,omitempty"`
		AnthropicVersion string    `json:"anthropic_version"`
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

	// Handle TopK from config
	if topK, ok := request.Config["top_k"].(int); ok {
		req.TopK = topK
	}

	return json.Marshal(req)
}

// ParseResponse parses an Anthropic response.
func (a *AnthropicAdapter) ParseResponse(body []byte, modelID string) (*clients.Response, error) {
	return parseAnthropicResponse(body, modelID)
}

// TitanAdapter handles Amazon Titan models.
type TitanAdapter struct{}

// BuildRequest builds a Titan-formatted request.
func (t *TitanAdapter) BuildRequest(request *clients.Request) ([]byte, error) {
	return buildTitanRequest(request)
}

// ParseResponse parses a Titan response.
func (t *TitanAdapter) ParseResponse(body []byte, modelID string) (*clients.Response, error) {
	return parseTitanResponse(body, modelID)
}

// LlamaAdapter handles Meta Llama models.
type LlamaAdapter struct{}

// BuildRequest builds a Llama-formatted request.
func (l *LlamaAdapter) BuildRequest(request *clients.Request) ([]byte, error) {
	return buildLlamaRequest(request)
}

// ParseResponse parses a Llama response.
func (l *LlamaAdapter) ParseResponse(body []byte, modelID string) (*clients.Response, error) {
	return parseLlamaResponse(body, modelID)
}

// AI21Adapter handles AI21 Labs Jurassic models.
type AI21Adapter struct{}

// BuildRequest builds an AI21-formatted request.
func (a *AI21Adapter) BuildRequest(request *clients.Request) ([]byte, error) {
	type Request struct {
		Prompt        string   `json:"prompt"`
		Temperature   float64  `json:"temperature,omitempty"`
		TopP          float64  `json:"topP,omitempty"`
		MaxTokens     int      `json:"maxTokens,omitempty"`
		StopSequences []string `json:"stopSequences,omitempty"`
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
		Prompt:        prompt,
		Temperature:   request.Temperature,
		TopP:          request.TopP,
		MaxTokens:     maxTokens,
		StopSequences: request.StopSequences,
	}

	return json.Marshal(req)
}

// ParseResponse parses an AI21 response.
func (a *AI21Adapter) ParseResponse(body []byte, modelID string) (*clients.Response, error) {
	type Completion struct {
		Data struct {
			Text string `json:"text"`
		} `json:"data"`
		FinishReason string `json:"finishReason"`
	}

	type Response struct {
		ID          string       `json:"id"`
		Completions []Completion `json:"completions"`
	}

	var resp Response
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(resp.Completions) == 0 {
		return nil, fmt.Errorf("no completions in response")
	}

	completion := resp.Completions[0]

	return &clients.Response{
		ID:    resp.ID,
		Model: modelID,
		Choices: []clients.Choice{
			{
				Index:        0,
				FinishReason: completion.FinishReason,
				Text:         completion.Data.Text,
			},
		},
		Usage: clients.Usage{
			// AI21 doesn't provide token counts in all responses
			PromptTokens:     0,
			CompletionTokens: 0,
			TotalTokens:      0,
		},
	}, nil
}

// CohereAdapter handles Cohere Command models.
type CohereAdapter struct{}

// BuildRequest builds a Cohere-formatted request.
func (c *CohereAdapter) BuildRequest(request *clients.Request) ([]byte, error) {
	type Request struct {
		Prompt        string   `json:"prompt"`
		Temperature   float64  `json:"temperature,omitempty"`
		P             float64  `json:"p,omitempty"`
		MaxTokens     int      `json:"max_tokens,omitempty"`
		StopSequences []string `json:"stop_sequences,omitempty"`
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
		Prompt:        prompt,
		Temperature:   request.Temperature,
		P:             request.TopP,
		MaxTokens:     maxTokens,
		StopSequences: request.StopSequences,
	}

	return json.Marshal(req)
}

// ParseResponse parses a Cohere response.
func (c *CohereAdapter) ParseResponse(body []byte, modelID string) (*clients.Response, error) {
	type Generation struct {
		Text         string `json:"text"`
		FinishReason string `json:"finish_reason"`
	}

	type Response struct {
		ID          string       `json:"id"`
		Generations []Generation `json:"generations"`
	}

	var resp Response
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(resp.Generations) == 0 {
		return nil, fmt.Errorf("no generations in response")
	}

	generation := resp.Generations[0]

	return &clients.Response{
		ID:    resp.ID,
		Model: modelID,
		Choices: []clients.Choice{
			{
				Index:        0,
				FinishReason: generation.FinishReason,
				Text:         generation.Text,
			},
		},
		Usage: clients.Usage{
			PromptTokens:     0,
			CompletionTokens: 0,
			TotalTokens:      0,
		},
	}, nil
}

// GetModelAdapter returns the appropriate adapter for a model.
func GetModelAdapter(modelID string) ModelAdapter {
	provider := getModelProvider(modelID)

	switch provider {
	case "anthropic":
		return &AnthropicAdapter{}
	case "titan":
		return &TitanAdapter{}
	case "llama":
		return &LlamaAdapter{}
	case "ai21":
		return &AI21Adapter{}
	case "cohere":
		return &CohereAdapter{}
	default:
		// Default to Anthropic adapter
		return &AnthropicAdapter{}
	}
}
