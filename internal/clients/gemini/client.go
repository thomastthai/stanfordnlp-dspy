// Package gemini provides a Google Gemini API client implementation.
package gemini

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
	defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"
	defaultTimeout = 60 * time.Second
)

// Client is a Google Gemini API client.
type Client struct {
	apiKey     string
	baseURL    string
	httpClient *retryablehttp.Client
}

// ClientOptions configures the Gemini client.
type ClientOptions struct {
	APIKey  string
	BaseURL string
	Timeout time.Duration
}

// NewClient creates a new Gemini client.
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
	}, nil
}

// GenerateContentRequest represents a Gemini API request.
type GenerateContentRequest struct {
	Contents         []Content         `json:"contents"`
	GenerationConfig *GenerationConfig `json:"generationConfig,omitempty"`
	SafetySettings   []SafetySetting   `json:"safetySettings,omitempty"`
}

// Content represents message content.
type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts"`
}

// Part represents a part of the content.
type Part struct {
	Text string `json:"text"`
}

// GenerationConfig configures generation parameters.
type GenerationConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	TopK            *int     `json:"topK,omitempty"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
	CandidateCount  *int     `json:"candidateCount,omitempty"`
}

// SafetySetting configures safety settings.
type SafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

// GenerateContentResponse represents a Gemini API response.
type GenerateContentResponse struct {
	Candidates     []Candidate     `json:"candidates"`
	PromptFeedback *PromptFeedback `json:"promptFeedback,omitempty"`
	UsageMetadata  *UsageMetadata  `json:"usageMetadata,omitempty"`
}

// Candidate represents a generation candidate.
type Candidate struct {
	Content       Content  `json:"content"`
	FinishReason  string   `json:"finishReason"`
	Index         int      `json:"index"`
	SafetyRatings []Rating `json:"safetyRatings,omitempty"`
}

// Rating represents a safety rating.
type Rating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"`
}

// PromptFeedback provides feedback about the prompt.
type PromptFeedback struct {
	SafetyRatings []Rating `json:"safetyRatings,omitempty"`
}

// UsageMetadata contains token usage information.
type UsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// Call sends a request to the Gemini API.
func (c *Client) Call(ctx context.Context, request *clients.Request, model string) (*clients.Response, error) {
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
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Build URL
	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s", c.baseURL, model, c.apiKey)

	// Create HTTP request
	httpReq, err := retryablehttp.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// Make the API call
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Gemini API call failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Gemini API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var geminiResp GenerateContentResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Convert to our response format
	response := &clients.Response{
		Model:   model,
		Choices: make([]clients.Choice, 0, len(geminiResp.Candidates)),
	}

	// Add usage information if available
	if geminiResp.UsageMetadata != nil {
		response.Usage = clients.Usage{
			PromptTokens:     geminiResp.UsageMetadata.PromptTokenCount,
			CompletionTokens: geminiResp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      geminiResp.UsageMetadata.TotalTokenCount,
		}
	}

	for _, candidate := range geminiResp.Candidates {
		// Extract text from parts
		var text string
		for _, part := range candidate.Content.Parts {
			text += part.Text
		}

		response.Choices = append(response.Choices, clients.Choice{
			Index:        candidate.Index,
			FinishReason: candidate.FinishReason,
			Message: clients.Message{
				Role:    "assistant",
				Content: text,
			},
		})
	}

	return response, nil
}
