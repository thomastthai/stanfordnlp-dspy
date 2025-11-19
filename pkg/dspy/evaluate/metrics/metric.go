// Package metrics provides evaluation metrics for DSPy.
package metrics

// Metric is a function that computes a score between prediction and reference.
type Metric interface {
	// Compute calculates the metric for a single prediction-reference pair
	Compute(prediction string, reference string) float64
	
	// ComputeBatch calculates the metric for multiple pairs
	ComputeBatch(predictions, references []string) []float64
	
	// Name returns the metric name
	Name() string
}

// BaseMetric provides common functionality for metrics.
type BaseMetric struct {
	name string
}

// Name returns the metric name.
func (m *BaseMetric) Name() string {
	return m.name
}

// ComputeBatch provides default batch computation.
func (m *BaseMetric) ComputeBatch(predictions, references []string, computeFn func(string, string) float64) []float64 {
	if len(predictions) != len(references) {
		return nil
	}
	
	scores := make([]float64, len(predictions))
	for i := range predictions {
		scores[i] = computeFn(predictions[i], references[i])
	}
	
	return scores
}

// min returns the minimum of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the maximum of two integers.
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
