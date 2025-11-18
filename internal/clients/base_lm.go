// Package clients provides language model client implementations.
package clients

import (
	"context"
)

// BaseLM defines the interface that all language model clients must implement.
// This follows the provider pattern similar to Terraform.
type BaseLM interface {
	// Call sends a request to the language model and returns the response.
	Call(ctx context.Context, request *Request) (*Response, error)

	// CallBatch sends multiple requests in a batch.
	CallBatch(ctx context.Context, requests []*Request) ([]*Response, error)

	// Name returns the name of the language model.
	Name() string

	// Provider returns the provider name (e.g., "openai", "anthropic").
	Provider() string
}

// Request represents a request to a language model.
type Request struct {
	// Messages is the list of messages to send (for chat-based models)
	Messages []Message

	// Prompt is the text prompt (for completion-based models)
	Prompt string

	// Config contains model-specific configuration
	Config map[string]interface{}

	// Temperature for sampling
	Temperature float64

	// MaxTokens is the maximum number of tokens to generate
	MaxTokens int

	// TopP for nucleus sampling
	TopP float64

	// StopSequences are sequences that stop generation
	StopSequences []string

	// N is the number of completions to generate
	N int

	// Stream indicates if the response should be streamed
	Stream bool

	// Metadata contains additional request metadata
	Metadata map[string]interface{}
}

// Message represents a message in a conversation.
type Message struct {
	// Role is the role of the message sender (e.g., "user", "assistant", "system")
	Role string

	// Content is the message content
	Content string

	// Name is an optional name for the message sender
	Name string

	// FunctionCall contains function call information (if applicable)
	FunctionCall *FunctionCall

	// ToolCalls contains tool call information (if applicable)
	ToolCalls []ToolCall
}

// FunctionCall represents a function call request from the model.
type FunctionCall struct {
	Name      string
	Arguments string
}

// ToolCall represents a tool call request from the model.
type ToolCall struct {
	ID       string
	Type     string
	Function FunctionCall
}

// Response represents a response from a language model.
type Response struct {
	// Choices are the generated completions
	Choices []Choice

	// Usage contains token usage information
	Usage Usage

	// Model is the model that generated the response
	Model string

	// ID is the response ID
	ID string

	// Metadata contains additional response metadata
	Metadata map[string]interface{}
}

// Choice represents a single completion choice.
type Choice struct {
	// Message is the generated message (for chat models)
	Message Message

	// Text is the generated text (for completion models)
	Text string

	// Index is the choice index
	Index int

	// FinishReason indicates why generation stopped
	FinishReason string

	// Logprobs contains log probabilities (if requested)
	Logprobs interface{}
}

// Usage contains token usage information.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// NewRequest creates a new Request with default values.
func NewRequest() *Request {
	return &Request{
		Messages:      []Message{},
		Config:        make(map[string]interface{}),
		Temperature:   0.0,
		MaxTokens:     1000,
		TopP:          1.0,
		StopSequences: []string{},
		N:             1,
		Stream:        false,
		Metadata:      make(map[string]interface{}),
	}
}

// WithMessages adds messages to the request.
func (r *Request) WithMessages(messages ...Message) *Request {
	r.Messages = append(r.Messages, messages...)
	return r
}

// WithPrompt sets the prompt for the request.
func (r *Request) WithPrompt(prompt string) *Request {
	r.Prompt = prompt
	return r
}

// WithTemperature sets the temperature.
func (r *Request) WithTemperature(temp float64) *Request {
	r.Temperature = temp
	return r
}

// WithMaxTokens sets the max tokens.
func (r *Request) WithMaxTokens(max int) *Request {
	r.MaxTokens = max
	return r
}

// WithConfig sets a config value.
func (r *Request) WithConfig(key string, value interface{}) *Request {
	r.Config[key] = value
	return r
}

// NewMessage creates a new Message.
func NewMessage(role, content string) Message {
	return Message{
		Role:    role,
		Content: content,
	}
}
