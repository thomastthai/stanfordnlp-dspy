package evaluate

import (
	"fmt"
	"strings"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// ExactMatch checks if the prediction exactly matches the expected output.
func ExactMatch(field string) Metric {
	return func(example *primitives.Example, prediction *primitives.Prediction) float64 {
		expected, ok := example.Get(field)
		if !ok {
			return 0.0
		}

		predicted, ok := prediction.Get(field)
		if !ok {
			return 0.0
		}

		if expected == predicted {
			return 1.0
		}
		return 0.0
	}
}

// ContainsMatch checks if the prediction contains the expected output.
func ContainsMatch(field string) Metric {
	return func(example *primitives.Example, prediction *primitives.Prediction) float64 {
		expected, ok := example.Get(field)
		if !ok {
			return 0.0
		}

		predicted, ok := prediction.Get(field)
		if !ok {
			return 0.0
		}

		expectedStr := strings.ToLower(fmt.Sprint(expected))
		predictedStr := strings.ToLower(fmt.Sprint(predicted))

		if strings.Contains(predictedStr, expectedStr) {
			return 1.0
		}
		return 0.0
	}
}

// F1Score computes the F1 score for token overlap.
func F1Score(field string) Metric {
	return func(example *primitives.Example, prediction *primitives.Prediction) float64 {
		expected, ok := example.Get(field)
		if !ok {
			return 0.0
		}

		predicted, ok := prediction.Get(field)
		if !ok {
			return 0.0
		}

		expectedStr := strings.ToLower(fmt.Sprint(expected))
		predictedStr := strings.ToLower(fmt.Sprint(predicted))

		// Tokenize (simple whitespace split)
		expectedTokens := strings.Fields(expectedStr)
		predictedTokens := strings.Fields(predictedStr)

		if len(expectedTokens) == 0 || len(predictedTokens) == 0 {
			return 0.0
		}

		// Count overlaps
		expectedSet := make(map[string]int)
		for _, token := range expectedTokens {
			expectedSet[token]++
		}

		predictedSet := make(map[string]int)
		for _, token := range predictedTokens {
			predictedSet[token]++
		}

		overlap := 0
		for token, count := range predictedSet {
			if expectedCount, ok := expectedSet[token]; ok {
				overlap += min(count, expectedCount)
			}
		}

		if overlap == 0 {
			return 0.0
		}

		precision := float64(overlap) / float64(len(predictedTokens))
		recall := float64(overlap) / float64(len(expectedTokens))

		f1 := 2 * precision * recall / (precision + recall)
		return f1
	}
}

// min returns the minimum of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
