package outbound

import (
	"context"
)

// LanguageModelPort defines the interface for language model providers.
type LanguageModelPort interface {
	Generate(ctx context.Context, request *GenerateRequest) (*GenerateResponse, error)
	GenerateBatch(ctx context.Context, requests []*GenerateRequest) ([]*GenerateResponse, error)
	Name() string
	Provider() string
}

type GenerateRequest struct {
	Prompt      string
	MaxTokens   int
	Temperature float64
	TopP        float64
	Stop        []string
}

type GenerateResponse struct {
	Text         string
	FinishReason string
	Usage        TokenUsage
}

type TokenUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}
