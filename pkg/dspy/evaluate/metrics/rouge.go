package metrics

import (
	"strings"
)

// ROUGEMetric computes ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores.
// Reference: Lin 2004
type ROUGEMetric struct {
	BaseMetric
	variant string // "rouge-n", "rouge-l", "rouge-w"
	n       int    // For ROUGE-N
	beta    float64 // F-measure beta (typically 1.2 for ROUGE-L)
}

// NewROUGEN creates a new ROUGE-N metric.
func NewROUGEN(n int) *ROUGEMetric {
	return &ROUGEMetric{
		BaseMetric: BaseMetric{name: "rouge-n"},
		variant:    "rouge-n",
		n:          n,
		beta:       1.2,
	}
}

// NewROUGEL creates a new ROUGE-L metric (Longest Common Subsequence).
func NewROUGEL() *ROUGEMetric {
	return &ROUGEMetric{
		BaseMetric: BaseMetric{name: "rouge-l"},
		variant:    "rouge-l",
		beta:       1.2,
	}
}

// Compute implements Metric.Compute.
func (m *ROUGEMetric) Compute(prediction string, reference string) float64 {
	predTokens := strings.Fields(strings.ToLower(prediction))
	refTokens := strings.Fields(strings.ToLower(reference))
	
	switch m.variant {
	case "rouge-n":
		return m.computeROUGEN(predTokens, refTokens)
	case "rouge-l":
		return m.computeROUGEL(predTokens, refTokens)
	default:
		return 0.0
	}
}

// ComputeBatch implements Metric.ComputeBatch.
func (m *ROUGEMetric) ComputeBatch(predictions, references []string) []float64 {
	return m.BaseMetric.ComputeBatch(predictions, references, m.Compute)
}

// computeROUGEN calculates ROUGE-N score.
func (m *ROUGEMetric) computeROUGEN(predTokens, refTokens []string) float64 {
	if len(refTokens) == 0 {
		return 0.0
	}
	
	predNgrams := m.getNgrams(predTokens, m.n)
	refNgrams := m.getNgrams(refTokens, m.n)
	
	if len(refNgrams) == 0 {
		return 0.0
	}
	
	// Count overlapping n-grams
	overlap := 0
	for ngram, refCount := range refNgrams {
		predCount := predNgrams[ngram]
		overlap += min(predCount, refCount)
	}
	
	// Calculate precision, recall, and F-measure
	precision := 0.0
	if len(predNgrams) > 0 {
		precision = float64(overlap) / float64(len(predNgrams))
	}
	
	recall := float64(overlap) / float64(len(refNgrams))
	
	if precision+recall == 0 {
		return 0.0
	}
	
	// F-measure with beta
	betaSq := m.beta * m.beta
	fScore := (1 + betaSq) * precision * recall / (betaSq*precision + recall)
	
	return fScore
}

// computeROUGEL calculates ROUGE-L score using LCS.
func (m *ROUGEMetric) computeROUGEL(predTokens, refTokens []string) float64 {
	if len(predTokens) == 0 || len(refTokens) == 0 {
		return 0.0
	}
	
	lcsLen := m.lcs(predTokens, refTokens)
	
	// Calculate precision and recall based on LCS
	precision := float64(lcsLen) / float64(len(predTokens))
	recall := float64(lcsLen) / float64(len(refTokens))
	
	if precision+recall == 0 {
		return 0.0
	}
	
	// F-measure with beta
	betaSq := m.beta * m.beta
	fScore := (1 + betaSq) * precision * recall / (betaSq*precision + recall)
	
	return fScore
}

// getNgrams extracts n-grams from tokens.
func (m *ROUGEMetric) getNgrams(tokens []string, n int) map[string]int {
	ngrams := make(map[string]int)
	
	if len(tokens) < n {
		return ngrams
	}
	
	for i := 0; i <= len(tokens)-n; i++ {
		ngram := strings.Join(tokens[i:i+n], " ")
		ngrams[ngram]++
	}
	
	return ngrams
}

// lcs computes the length of the longest common subsequence.
func (m *ROUGEMetric) lcs(seq1, seq2 []string) int {
	m1 := len(seq1)
	m2 := len(seq2)
	
	// Create DP table
	dp := make([][]int, m1+1)
	for i := range dp {
		dp[i] = make([]int, m2+1)
	}
	
	// Fill DP table
	for i := 1; i <= m1; i++ {
		for j := 1; j <= m2; j++ {
			if seq1[i-1] == seq2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	
	return dp[m1][m2]
}

// ROUGEScores holds all ROUGE scores.
type ROUGEScores struct {
	ROUGE1 float64
	ROUGE2 float64
	ROUGEL float64
}

// ComputeAllROUGE computes ROUGE-1, ROUGE-2, and ROUGE-L.
func ComputeAllROUGE(prediction string, reference string) ROUGEScores {
	rouge1 := NewROUGEN(1)
	rouge2 := NewROUGEN(2)
	rougeL := NewROUGEL()
	
	return ROUGEScores{
		ROUGE1: rouge1.Compute(prediction, reference),
		ROUGE2: rouge2.Compute(prediction, reference),
		ROUGEL: rougeL.Compute(prediction, reference),
	}
}
