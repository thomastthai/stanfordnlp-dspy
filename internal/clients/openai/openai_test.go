package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stanfordnlp/dspy/internal/clients"
)

func TestNewClient(t *testing.T) {
	// Test with API key
	client, err := NewClient(ClientOptions{
		APIKey: "test-key",
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	if client.apiKey != "test-key" {
		t.Errorf("Expected API key 'test-key', got '%s'", client.apiKey)
	}
	if client.baseURL != defaultBaseURL {
		t.Errorf("Expected base URL '%s', got '%s'", defaultBaseURL, client.baseURL)
	}

	// Test without API key
	_, err = NewClient(ClientOptions{})
	if err == nil {
		t.Error("Expected error when creating client without API key")
	}
}

func TestChatCompletion(t *testing.T) {
	// Create mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != "POST" {
			t.Errorf("Expected POST request, got %s", r.Method)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("Expected Authorization header 'Bearer test-key', got '%s'", r.Header.Get("Authorization"))
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("Expected Content-Type 'application/json', got '%s'", r.Header.Get("Content-Type"))
		}

		// Send mock response
		resp := ChatCompletionResponse{
			ID:      "chatcmpl-123",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "gpt-4o-mini",
			Choices: []Choice{
				{
					Index: 0,
					Message: ChatMessage{
						Role:    "assistant",
						Content: "Hello! How can I help you today?",
					},
					FinishReason: "stop",
				},
			},
			Usage: Usage{
				PromptTokens:     10,
				CompletionTokens: 9,
				TotalTokens:      19,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Create client with mock server
	client, err := NewClient(ClientOptions{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Make request
	ctx := context.Background()
	req := ChatCompletionRequest{
		Model: "gpt-4o-mini",
		Messages: []ChatMessage{
			{Role: "user", Content: "Hello!"},
		},
	}

	resp, err := client.ChatCompletion(ctx, req)
	if err != nil {
		t.Fatalf("ChatCompletion failed: %v", err)
	}

	// Verify response
	if resp.ID != "chatcmpl-123" {
		t.Errorf("Expected ID 'chatcmpl-123', got '%s'", resp.ID)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("Expected 1 choice, got %d", len(resp.Choices))
	}
	if resp.Choices[0].Message.Content != "Hello! How can I help you today?" {
		t.Errorf("Unexpected message content: %s", resp.Choices[0].Message.Content)
	}
	if resp.Usage.TotalTokens != 19 {
		t.Errorf("Expected 19 total tokens, got %d", resp.Usage.TotalTokens)
	}
}

func TestChatCompletionError(t *testing.T) {
	// Create mock server that returns an error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		errResp := ErrorResponse{
			Error: ErrorDetails{
				Message: "Invalid API key",
				Type:    "invalid_request_error",
				Code:    "invalid_api_key",
			},
		}
		json.NewEncoder(w).Encode(errResp)
	}))
	defer server.Close()

	client, err := NewClient(ClientOptions{
		APIKey:  "invalid-key",
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	ctx := context.Background()
	req := ChatCompletionRequest{
		Model: "gpt-4o-mini",
		Messages: []ChatMessage{
			{Role: "user", Content: "Hello!"},
		},
	}

	_, err = client.ChatCompletion(ctx, req)
	if err == nil {
		t.Error("Expected error for invalid API key")
	}
}

func TestCall_BaseLM(t *testing.T) {
	// Create mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := ChatCompletionResponse{
			ID:      "chatcmpl-456",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "gpt-4o-mini",
			Choices: []Choice{
				{
					Index: 0,
					Message: ChatMessage{
						Role:    "assistant",
						Content: "Test response",
					},
					FinishReason: "stop",
				},
			},
			Usage: Usage{
				PromptTokens:     5,
				CompletionTokens: 2,
				TotalTokens:      7,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client, err := NewClient(ClientOptions{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Test Call method
	ctx := context.Background()
	req := clients.NewRequest().
		WithMessages(clients.NewMessage("user", "Hello")).
		WithTemperature(0.7).
		WithMaxTokens(100)

	resp, err := client.Call(ctx, req, "gpt-4o-mini")
	if err != nil {
		t.Fatalf("Call failed: %v", err)
	}

	if len(resp.Choices) != 1 {
		t.Fatalf("Expected 1 choice, got %d", len(resp.Choices))
	}
	if resp.Choices[0].Text != "Test response" {
		t.Errorf("Unexpected response text: %s", resp.Choices[0].Text)
	}
	if resp.Usage.TotalTokens != 7 {
		t.Errorf("Expected 7 total tokens, got %d", resp.Usage.TotalTokens)
	}
}

func TestProvider_Create(t *testing.T) {
	provider := &Provider{}

	// Test with valid config
	config := map[string]interface{}{
		"model":   "gpt-4o-mini",
		"api_key": "test-key",
	}

	lm, err := provider.Create(config)
	if err != nil {
		t.Fatalf("Failed to create LM: %v", err)
	}

	if lm.Name() != "gpt-4o-mini" {
		t.Errorf("Expected model name 'gpt-4o-mini', got '%s'", lm.Name())
	}
	if lm.Provider() != "openai" {
		t.Errorf("Expected provider 'openai', got '%s'", lm.Provider())
	}

	// Test with missing model
	invalidConfig := map[string]interface{}{
		"api_key": "test-key",
	}
	_, err = provider.Create(invalidConfig)
	if err == nil {
		t.Error("Expected error when model is missing")
	}

	// Test with unsupported model
	unsupportedConfig := map[string]interface{}{
		"model":   "invalid-model",
		"api_key": "test-key",
	}
	_, err = provider.Create(unsupportedConfig)
	if err == nil {
		t.Error("Expected error for unsupported model")
	}
}

func TestProvider_SupportedModels(t *testing.T) {
	provider := &Provider{}
	models := provider.SupportedModels()

	if len(models) == 0 {
		t.Error("Expected non-empty list of supported models")
	}

	// Check for some expected models
	expectedModels := []string{"gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"}
	for _, expected := range expectedModels {
		found := false
		for _, model := range models {
			if model == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected model '%s' not found in supported models", expected)
		}
	}
}

func TestIsReasoningModel(t *testing.T) {
	tests := []struct {
		model    string
		expected bool
	}{
		{"o1", true},
		{"o1-preview", true},
		{"o1-mini", true},
		{"o3", true},
		{"o3-mini", true},
		{"gpt-4o", false},
		{"gpt-4o-mini", false},
		{"gpt-3.5-turbo", false},
	}

	for _, tt := range tests {
		result := IsReasoningModel(tt.model)
		if result != tt.expected {
			t.Errorf("IsReasoningModel(%s) = %v, expected %v", tt.model, result, tt.expected)
		}
	}
}

func TestLM_CallBatch(t *testing.T) {
	// Create mock server
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		resp := ChatCompletionResponse{
			ID:      "chatcmpl-batch",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "gpt-4o-mini",
			Choices: []Choice{
				{
					Index: 0,
					Message: ChatMessage{
						Role:    "assistant",
						Content: "Batch response",
					},
					FinishReason: "stop",
				},
			},
			Usage: Usage{
				PromptTokens:     5,
				CompletionTokens: 2,
				TotalTokens:      7,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client, err := NewClient(ClientOptions{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	lm := &LM{
		client: client,
		model:  "gpt-4o-mini",
	}

	// Test batch call
	ctx := context.Background()
	requests := []*clients.Request{
		clients.NewRequest().WithMessages(clients.NewMessage("user", "Hello 1")),
		clients.NewRequest().WithMessages(clients.NewMessage("user", "Hello 2")),
		clients.NewRequest().WithMessages(clients.NewMessage("user", "Hello 3")),
	}

	responses, err := lm.CallBatch(ctx, requests)
	if err != nil {
		t.Fatalf("CallBatch failed: %v", err)
	}

	if len(responses) != 3 {
		t.Fatalf("Expected 3 responses, got %d", len(responses))
	}
	if callCount != 3 {
		t.Errorf("Expected 3 API calls, got %d", callCount)
	}

	for i, resp := range responses {
		if len(resp.Choices) != 1 {
			t.Errorf("Response %d: expected 1 choice, got %d", i, len(resp.Choices))
		}
		if resp.Choices[0].Text != "Batch response" {
			t.Errorf("Response %d: unexpected text: %s", i, resp.Choices[0].Text)
		}
	}
}
