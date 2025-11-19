// Package finetune provides fine-tuning integration for DSPy.
package finetune

import (
	"context"
	"time"
)

// FineTuner is the interface for fine-tuning language models.
type FineTuner interface {
	// PrepareData prepares training data in the required format
	PrepareData(examples []Example) (string, error)
	
	// StartJob starts a fine-tuning job
	StartJob(ctx context.Context, config FineTuneConfig) (string, error)
	
	// GetStatus retrieves the status of a fine-tuning job
	GetStatus(ctx context.Context, jobID string) (JobStatus, error)
	
	// Cancel cancels a fine-tuning job
	Cancel(ctx context.Context, jobID string) error
	
	// GetModel retrieves the fine-tuned model ID
	GetModel(ctx context.Context, jobID string) (string, error)
}

// Example represents a training example for fine-tuning.
type Example struct {
	Messages []Message          `json:"messages"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// Message represents a message in a conversation.
type Message struct {
	Role    string `json:"role"`    // "system", "user", "assistant"
	Content string `json:"content"`
}

// FineTuneConfig configures a fine-tuning job.
type FineTuneConfig struct {
	BaseModel      string
	TrainingFile   string
	ValidationFile string
	Epochs         int
	BatchSize      int
	LearningRate   float64
	Suffix         string
	Seed           int64
}

// JobStatus represents the status of a fine-tuning job.
type JobStatus struct {
	ID           string
	Status       string // "pending", "running", "succeeded", "failed", "cancelled"
	CreatedAt    time.Time
	FinishedAt   time.Time
	Model        string
	TrainedTokens int
	Error        string
	Metrics      map[string]float64
}

// DefaultFineTuneConfig returns default fine-tuning configuration.
func DefaultFineTuneConfig(baseModel string) FineTuneConfig {
	return FineTuneConfig{
		BaseModel:    baseModel,
		Epochs:       3,
		BatchSize:    1,
		LearningRate: 0.0001,
	}
}
