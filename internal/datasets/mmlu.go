package datasets

import (
	"context"
	"fmt"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// MMLU represents the Massive Multitask Language Understanding dataset.
type MMLU struct {
	*BaseDataset
	subjects []string
}

// MMLUOptions extends DatasetOptions with MMLU-specific options.
type MMLUOptions struct {
	DatasetOptions
	Subjects []string // Specific subjects to load, empty means all
}

// DefaultMMLUOptions returns default MMLU options.
func DefaultMMLUOptions() MMLUOptions {
	return MMLUOptions{
		DatasetOptions: DefaultDatasetOptions(),
		Subjects:       []string{}, // All subjects
	}
}

// NewMMLU creates a new MMLU dataset loader.
func NewMMLU(ctx context.Context, opts MMLUOptions) (*MMLU, error) {
	base := NewBaseDataset("mmlu", opts.DatasetOptions)
	dataset := &MMLU{
		BaseDataset: base,
		subjects:    opts.Subjects,
	}
	
	// Load dataset
	if err := dataset.load(ctx); err != nil {
		return nil, err
	}
	
	return dataset, nil
}

func (m *MMLU) load(ctx context.Context) error {
	// Placeholder implementation
	return fmt.Errorf("MMLU requires pre-downloaded data or HuggingFace API integration")
}

// MMLUExample represents a single MMLU question.
type MMLUExample struct {
	Question string   `json:"question"`
	Choices  []string `json:"choices"` // A, B, C, D
	Answer   string   `json:"answer"`  // Correct choice letter
	Subject  string   `json:"subject"`
}

// ConvertMMLUExample converts a raw MMLU example to a DSPy Example.
func ConvertMMLUExample(raw MMLUExample) *primitives.Example {
	data := map[string]interface{}{
		"question": raw.Question,
		"choices":  raw.Choices,
		"answer":   raw.Answer,
		"subject":  raw.Subject,
	}
	
	ex := primitives.NewExample(nil, data)
	return ex.WithInputs("question", "choices")
}

// MMLUMetric evaluates if the prediction matches the gold answer.
func MMLUMetric(gold, pred *primitives.Example) bool {
	goldAnswer, ok := gold.Get("answer")
	if !ok {
		return false
	}
	
	predAnswer, ok := pred.Get("answer")
	if !ok {
		return false
	}
	
	goldStr, ok := goldAnswer.(string)
	if !ok {
		return false
	}
	
	predStr, ok := predAnswer.(string)
	if !ok {
		return false
	}
	
	return goldStr == predStr
}

// Subjects returns the list of all MMLU subjects.
func Subjects() []string {
	return []string{
		"abstract_algebra", "anatomy", "astronomy", "business_ethics",
		"clinical_knowledge", "college_biology", "college_chemistry",
		"college_computer_science", "college_mathematics", "college_medicine",
		"college_physics", "computer_security", "conceptual_physics",
		"econometrics", "electrical_engineering", "elementary_mathematics",
		"formal_logic", "global_facts", "high_school_biology",
		"high_school_chemistry", "high_school_computer_science",
		"high_school_european_history", "high_school_geography",
		"high_school_government_and_politics", "high_school_macroeconomics",
		"high_school_mathematics", "high_school_microeconomics",
		"high_school_physics", "high_school_psychology",
		"high_school_statistics", "high_school_us_history",
		"high_school_world_history", "human_aging", "human_sexuality",
		"international_law", "jurisprudence", "logical_fallacies",
		"machine_learning", "management", "marketing", "medical_genetics",
		"miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
		"philosophy", "prehistory", "professional_accounting",
		"professional_law", "professional_medicine", "professional_psychology",
		"public_relations", "security_studies", "sociology", "us_foreign_policy",
		"virology", "world_religions",
	}
}
