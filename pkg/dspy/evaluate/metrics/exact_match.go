package metrics

import (
	"strings"
	"unicode"
)

// ExactMatchMetric computes exact match between prediction and reference.
type ExactMatchMetric struct {
	BaseMetric
	normalize bool
}

// NewExactMatch creates a new exact match metric.
func NewExactMatch(normalize bool) *ExactMatchMetric {
	return &ExactMatchMetric{
		BaseMetric: BaseMetric{name: "exact_match"},
		normalize:  normalize,
	}
}

// Compute implements Metric.Compute.
func (m *ExactMatchMetric) Compute(prediction string, reference string) float64 {
	pred := prediction
	ref := reference
	
	if m.normalize {
		pred = m.normalizeAnswer(pred)
		ref = m.normalizeAnswer(ref)
	}
	
	if pred == ref {
		return 1.0
	}
	return 0.0
}

// ComputeBatch implements Metric.ComputeBatch.
func (m *ExactMatchMetric) ComputeBatch(predictions, references []string) []float64 {
	return m.BaseMetric.ComputeBatch(predictions, references, m.Compute)
}

// normalizeAnswer normalizes text for comparison.
func (m *ExactMatchMetric) normalizeAnswer(text string) string {
	// Convert to lowercase
	text = strings.ToLower(text)
	
	// Remove punctuation
	text = strings.Map(func(r rune) rune {
		if unicode.IsPunct(r) {
			return -1
		}
		return r
	}, text)
	
	// Remove extra whitespace
	fields := strings.Fields(text)
	return strings.Join(fields, " ")
}

// F1Metric computes token-level F1 score.
type F1Metric struct {
	BaseMetric
	normalize bool
}

// NewF1 creates a new F1 metric.
func NewF1(normalize bool) *F1Metric {
	return &F1Metric{
		BaseMetric: BaseMetric{name: "f1"},
		normalize:  normalize,
	}
}

// Compute implements Metric.Compute.
func (m *F1Metric) Compute(prediction string, reference string) float64 {
	pred := prediction
	ref := reference
	
	if m.normalize {
		pred = m.normalizeAnswer(pred)
		ref = m.normalizeAnswer(ref)
	}
	
	// Tokenize
	predTokens := strings.Fields(pred)
	refTokens := strings.Fields(ref)
	
	if len(predTokens) == 0 || len(refTokens) == 0 {
		if len(predTokens) == 0 && len(refTokens) == 0 {
			return 1.0
		}
		return 0.0
	}
	
	// Count token overlaps
	predSet := make(map[string]int)
	for _, token := range predTokens {
		predSet[token]++
	}
	
	refSet := make(map[string]int)
	for _, token := range refTokens {
		refSet[token]++
	}
	
	overlap := 0
	for token, count := range predSet {
		if refCount, ok := refSet[token]; ok {
			overlap += min(count, refCount)
		}
	}
	
	if overlap == 0 {
		return 0.0
	}
	
	precision := float64(overlap) / float64(len(predTokens))
	recall := float64(overlap) / float64(len(refTokens))
	
	f1 := 2 * precision * recall / (precision + recall)
	return f1
}

// ComputeBatch implements Metric.ComputeBatch.
func (m *F1Metric) ComputeBatch(predictions, references []string) []float64 {
	return m.BaseMetric.ComputeBatch(predictions, references, m.Compute)
}

// normalizeAnswer normalizes text for comparison.
func (m *F1Metric) normalizeAnswer(text string) string {
	// Convert to lowercase
	text = strings.ToLower(text)
	
	// Remove punctuation
	text = strings.Map(func(r rune) rune {
		if unicode.IsPunct(r) {
			return -1
		}
		return r
	}, text)
	
	// Remove extra whitespace
	fields := strings.Fields(text)
	return strings.Join(fields, " ")
}
