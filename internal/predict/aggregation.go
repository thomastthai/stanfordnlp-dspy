// Package predict provides prediction modules that interact with language models.
package predict

import (
	"context"
	"fmt"
	"strings"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// Aggregation combines multiple predictions using voting strategies.
type Aggregation struct {
	*primitives.BaseModule

	// Strategy determines how to aggregate predictions
	// Supported: "majority", "weighted", "consensus"
	Strategy string

	// NormalizeFunc normalizes text for comparison
	NormalizeFunc func(string) string

	// Config contains additional configuration
	Config map[string]interface{}
}

// NewAggregation creates a new Aggregation module.
func NewAggregation(strategy string) *Aggregation {
	return &Aggregation{
		BaseModule:    primitives.NewBaseModule(),
		Strategy:      strategy,
		NormalizeFunc: DefaultNormalizeFunc,
		Config:        make(map[string]interface{}),
	}
}

// DefaultNormalizeFunc is the default text normalization function.
func DefaultNormalizeFunc(s string) string {
	s = strings.ToLower(s)
	s = strings.TrimSpace(s)
	return s
}

// Forward aggregates multiple predictions.
// Input should contain a "predictions" field with a slice of predictions to aggregate.
func (a *Aggregation) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Extract predictions from inputs
	predsInterface, ok := inputs["predictions"]
	if !ok {
		return nil, fmt.Errorf("missing 'predictions' field in inputs")
	}

	preds, ok := predsInterface.([]*primitives.Prediction)
	if !ok {
		return nil, fmt.Errorf("predictions must be a slice of *primitives.Prediction")
	}

	if len(preds) == 0 {
		return nil, fmt.Errorf("no predictions to aggregate")
	}

	// Get the field to aggregate (default to last output field)
	field := a.getFieldToAggregate(inputs)

	// Apply the aggregation strategy
	switch a.Strategy {
	case "majority":
		return a.majorityVote(preds, field)
	case "weighted":
		return a.weightedVote(preds, field)
	case "consensus":
		return a.consensusVote(preds, field)
	default:
		return nil, fmt.Errorf("unknown aggregation strategy: %s", a.Strategy)
	}
}

func (a *Aggregation) getFieldToAggregate(inputs map[string]interface{}) string {
	if fieldInterface, ok := inputs["field"]; ok {
		if field, ok := fieldInterface.(string); ok {
			return field
		}
	}
	return "" // Will use the first available field
}

// majorityVote returns the most common prediction.
func (a *Aggregation) majorityVote(preds []*primitives.Prediction, field string) (*primitives.Prediction, error) {
	// Count occurrences of each value
	counts := make(map[string]int)
	predMap := make(map[string]*primitives.Prediction)

	for _, pred := range preds {
		outputs := pred.Fields()

		// Get the value for the specified field or use first output
		var value string
		if field != "" {
			if v, ok := outputs[field]; ok {
				value = fmt.Sprintf("%v", v)
			}
		} else {
			// Use first available output
			for _, v := range outputs {
				value = fmt.Sprintf("%v", v)
				break
			}
		}

		// Normalize and count
		normalized := a.NormalizeFunc(value)
		if normalized != "" {
			counts[normalized]++
			if _, exists := predMap[normalized]; !exists {
				predMap[normalized] = pred
			}
		}
	}

	// Find the most common value
	var maxCount int
	var majorityValue string
	for value, count := range counts {
		if count > maxCount {
			maxCount = count
			majorityValue = value
		}
	}

	if majorityValue == "" {
		return preds[0], nil // Fallback to first prediction
	}

	// Return the first prediction with the majority value
	result := predMap[majorityValue]
	result.SetMetadata("majority_count", maxCount)
	result.SetMetadata("total_predictions", len(preds))
	result.SetMetadata("aggregation_strategy", "majority")

	return result, nil
}

// weightedVote returns a weighted combination of predictions.
func (a *Aggregation) weightedVote(preds []*primitives.Prediction, field string) (*primitives.Prediction, error) {
	// For now, implement as majority vote with weights from metadata
	// This can be extended to use confidence scores or other weights
	return a.majorityVote(preds, field)
}

// consensusVote requires consensus among predictions.
func (a *Aggregation) consensusVote(preds []*primitives.Prediction, field string) (*primitives.Prediction, error) {
	if len(preds) == 0 {
		return nil, fmt.Errorf("no predictions for consensus")
	}

	// Get the first prediction's value
	firstOutputs := preds[0].Fields()
	var firstValue string
	if field != "" {
		if v, ok := firstOutputs[field]; ok {
			firstValue = fmt.Sprintf("%v", v)
		}
	} else {
		for _, v := range firstOutputs {
			firstValue = fmt.Sprintf("%v", v)
			break
		}
	}

	normalizedFirst := a.NormalizeFunc(firstValue)

	// Check if all predictions agree
	for _, pred := range preds[1:] {
		outputs := pred.Fields()
		var value string
		if field != "" {
			if v, ok := outputs[field]; ok {
				value = fmt.Sprintf("%v", v)
			}
		} else {
			for _, v := range outputs {
				value = fmt.Sprintf("%v", v)
				break
			}
		}

		if a.NormalizeFunc(value) != normalizedFirst {
			return nil, fmt.Errorf("no consensus: predictions disagree")
		}
	}

	result := preds[0]
	result.SetMetadata("consensus", true)
	result.SetMetadata("total_predictions", len(preds))
	result.SetMetadata("aggregation_strategy", "consensus")

	return result, nil
}

// Copy creates a deep copy of the Aggregation module.
func (a *Aggregation) Copy() primitives.Module {
	newAgg := &Aggregation{
		BaseModule:    primitives.NewBaseModule(),
		Strategy:      a.Strategy,
		NormalizeFunc: a.NormalizeFunc,
		Config:        make(map[string]interface{}),
	}

	for k, v := range a.Config {
		newAgg.Config[k] = v
	}

	return newAgg
}

// NamedParameters returns all parameters in this module.
func (a *Aggregation) NamedParameters() []primitives.NamedParameter {
	return []primitives.NamedParameter{}
}
