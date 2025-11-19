package cost

// ModelPricing contains pricing information for a model.
type ModelPricing struct {
	PromptPrice     float64 // Price per 1M prompt tokens
	CompletionPrice float64 // Price per 1M completion tokens
}

// Model pricing (as of 2024 - should be updated regularly)
var modelPricing = map[string]ModelPricing{
	// OpenAI GPT-4
	"gpt-4": {
		PromptPrice:     30.0,
		CompletionPrice: 60.0,
	},
	"gpt-4-32k": {
		PromptPrice:     60.0,
		CompletionPrice: 120.0,
	},
	"gpt-4-turbo": {
		PromptPrice:     10.0,
		CompletionPrice: 30.0,
	},
	"gpt-4-turbo-preview": {
		PromptPrice:     10.0,
		CompletionPrice: 30.0,
	},
	
	// OpenAI GPT-3.5
	"gpt-3.5-turbo": {
		PromptPrice:     0.5,
		CompletionPrice: 1.5,
	},
	"gpt-3.5-turbo-16k": {
		PromptPrice:     3.0,
		CompletionPrice: 4.0,
	},
	
	// OpenAI GPT-4o
	"gpt-4o": {
		PromptPrice:     5.0,
		CompletionPrice: 15.0,
	},
	"gpt-4o-mini": {
		PromptPrice:     0.15,
		CompletionPrice: 0.6,
	},
	
	// Anthropic Claude
	"claude-3-opus": {
		PromptPrice:     15.0,
		CompletionPrice: 75.0,
	},
	"claude-3-sonnet": {
		PromptPrice:     3.0,
		CompletionPrice: 15.0,
	},
	"claude-3-haiku": {
		PromptPrice:     0.25,
		CompletionPrice: 1.25,
	},
	"claude-3-5-sonnet": {
		PromptPrice:     3.0,
		CompletionPrice: 15.0,
	},
	
	// OpenAI Embeddings
	"text-embedding-ada-002": {
		PromptPrice:     0.1,
		CompletionPrice: 0.0,
	},
	"text-embedding-3-small": {
		PromptPrice:     0.02,
		CompletionPrice: 0.0,
	},
	"text-embedding-3-large": {
		PromptPrice:     0.13,
		CompletionPrice: 0.0,
	},
	
	// Cohere
	"command": {
		PromptPrice:     1.0,
		CompletionPrice: 2.0,
	},
	"command-light": {
		PromptPrice:     0.3,
		CompletionPrice: 0.6,
	},
}

// CalculateCost calculates the cost for a given model and token usage.
func CalculateCost(model string, promptTokens, completionTokens int) float64 {
	pricing, ok := modelPricing[model]
	if !ok {
		// Default pricing if model not found
		pricing = ModelPricing{
			PromptPrice:     1.0,
			CompletionPrice: 2.0,
		}
	}
	
	// Convert tokens to millions and calculate cost
	promptCost := float64(promptTokens) / 1_000_000.0 * pricing.PromptPrice
	completionCost := float64(completionTokens) / 1_000_000.0 * pricing.CompletionPrice
	
	return promptCost + completionCost
}

// SetModelPricing sets or updates pricing for a model.
func SetModelPricing(model string, pricing ModelPricing) {
	modelPricing[model] = pricing
}

// GetModelPricing returns the pricing for a model.
func GetModelPricing(model string) (ModelPricing, bool) {
	pricing, ok := modelPricing[model]
	return pricing, ok
}

// EstimateCost estimates the cost for a given model and approximate token count.
func EstimateCost(model string, estimatedTokens int) float64 {
	// Assume 50/50 split between prompt and completion
	promptTokens := estimatedTokens / 2
	completionTokens := estimatedTokens / 2
	return CalculateCost(model, promptTokens, completionTokens)
}
