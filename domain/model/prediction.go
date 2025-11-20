package model

import "time"

// Prediction represents a prediction result from a module.
type Prediction struct {
	ID         string
	Inputs     map[string]Field
	Outputs    map[string]Field
	Reasoning  []string
	Confidence float64
	Metadata   Metadata
	CreatedAt  time.Time
}

type Metadata struct {
	ModelName   string
	Provider    string
	TokensUsed  int
	Latency     time.Duration
	Temperature float64
	MaxTokens   int
}
