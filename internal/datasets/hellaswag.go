package datasets

import (
	"context"
	"fmt"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// HellaSwag represents the HellaSwag commonsense reasoning dataset.
type HellaSwag struct {
	*BaseDataset
}

// NewHellaSwag creates a new HellaSwag dataset loader.
func NewHellaSwag(ctx context.Context, opts DatasetOptions) (*HellaSwag, error) {
	base := NewBaseDataset("hellaswag", opts)
	dataset := &HellaSwag{BaseDataset: base}
	
	if err := dataset.load(ctx); err != nil {
		return nil, err
	}
	
	return dataset, nil
}

func (h *HellaSwag) load(ctx context.Context) error {
	// Placeholder implementation
	return fmt.Errorf("HellaSwag requires pre-downloaded data or HuggingFace API integration")
}

// HellaSwagExample represents a single HellaSwag example.
type HellaSwagExample struct {
	Context     string   `json:"ctx"`
	Endings     []string `json:"endings"`
	Label       int      `json:"label"` // Index of correct ending
	ActivityLabel string `json:"activity_label"`
}

// ConvertHellaSwagExample converts a raw HellaSwag example to a DSPy Example.
func ConvertHellaSwagExample(raw HellaSwagExample) *primitives.Example {
	data := map[string]interface{}{
		"context":        raw.Context,
		"endings":        raw.Endings,
		"label":          raw.Label,
		"activity_label": raw.ActivityLabel,
	}
	
	ex := primitives.NewExample(nil, data)
	return ex.WithInputs("context", "endings")
}

// HellaSwagMetric evaluates if the prediction matches the gold label.
func HellaSwagMetric(gold, pred *primitives.Example) bool {
	goldLabel, ok := gold.Get("label")
	if !ok {
		return false
	}
	
	predLabel, ok := pred.Get("label")
	if !ok {
		return false
	}
	
	goldInt, ok := goldLabel.(int)
	if !ok {
		return false
	}
	
	predInt, ok := predLabel.(int)
	if !ok {
		return false
	}
	
	return goldInt == predInt
}
