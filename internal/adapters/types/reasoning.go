package types

import (
	"fmt"
)

// ReasoningConfig configures reasoning token behavior for models like o1/o3.
type ReasoningConfig struct {
	// MaxReasoningTokens is the maximum number of reasoning tokens to use
	MaxReasoningTokens int

	// ReasoningEffort controls the effort level (low, medium, high)
	ReasoningEffort string

	// ShowReasoningTrace indicates whether to return the reasoning trace
	ShowReasoningTrace bool
}

// ReasoningEffortLevel represents reasoning effort levels.
type ReasoningEffortLevel string

const (
	ReasoningEffortLow    ReasoningEffortLevel = "low"
	ReasoningEffortMedium ReasoningEffortLevel = "medium"
	ReasoningEffortHigh   ReasoningEffortLevel = "high"
)

// ReasoningResponse contains the reasoning output from a model.
type ReasoningResponse struct {
	// ReasoningContent is the internal reasoning/thinking process
	ReasoningContent string

	// FinalAnswer is the final answer after reasoning
	FinalAnswer string

	// ReasoningTokens is the number of tokens used for reasoning
	ReasoningTokens int

	// Steps contains individual reasoning steps if available
	Steps []ReasoningStep
}

// ReasoningStep represents a single step in the reasoning process.
type ReasoningStep struct {
	// StepNumber is the step index
	StepNumber int

	// Content is the content of this reasoning step
	Content string

	// Confidence is the confidence score for this step (0-1)
	Confidence float64
}

// NewReasoningConfig creates a new reasoning configuration.
func NewReasoningConfig(maxTokens int, effort ReasoningEffortLevel) *ReasoningConfig {
	return &ReasoningConfig{
		MaxReasoningTokens: maxTokens,
		ReasoningEffort:    string(effort),
		ShowReasoningTrace: true,
	}
}

// Validate validates the reasoning configuration.
func (c *ReasoningConfig) Validate() error {
	if c.MaxReasoningTokens < 0 {
		return fmt.Errorf("max reasoning tokens must be non-negative")
	}

	// Validate effort level
	validEfforts := map[string]bool{
		string(ReasoningEffortLow):    true,
		string(ReasoningEffortMedium): true,
		string(ReasoningEffortHigh):   true,
	}

	if c.ReasoningEffort != "" && !validEfforts[c.ReasoningEffort] {
		return fmt.Errorf("invalid reasoning effort: %s (must be low, medium, or high)", c.ReasoningEffort)
	}

	return nil
}

// ToAPIParams converts the reasoning config to API parameters.
func (c *ReasoningConfig) ToAPIParams() map[string]interface{} {
	params := make(map[string]interface{})

	if c.MaxReasoningTokens > 0 {
		params["max_completion_tokens"] = c.MaxReasoningTokens
	}

	if c.ReasoningEffort != "" {
		params["reasoning_effort"] = c.ReasoningEffort
	}

	return params
}

// ExtractReasoning extracts reasoning information from a response.
func ExtractReasoning(content string) *ReasoningResponse {
	// This is a placeholder implementation
	// In practice, you'd parse the response to extract reasoning traces
	return &ReasoningResponse{
		ReasoningContent: content,
		FinalAnswer:      content,
		Steps:            []ReasoningStep{},
	}
}

// SupportsReasoning checks if a model supports reasoning tokens.
func SupportsReasoning(model string) bool {
	// Models that support reasoning tokens (o1, o3 series)
	return len(model) >= 2 && (model[:2] == "o1" || model[:2] == "o3")
}
