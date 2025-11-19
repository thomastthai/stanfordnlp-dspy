package metrics

import (
	"math"
	"strings"
)

// BLEUMetric computes BLEU (Bilingual Evaluation Understudy) score.
// Reference: Papineni et al. 2002
type BLEUMetric struct {
	BaseMetric
	n              int     // n-gram size (1, 2, 3, 4)
	weights        []float64
	smoothing      bool
}

// NewBLEU creates a new BLEU metric.
// n specifies the maximum n-gram size (typically 4 for BLEU-4).
func NewBLEU(n int) *BLEUMetric {
	if n <= 0 {
		n = 4
	}
	
	// Uniform weights for each n-gram
	weights := make([]float64, n)
	for i := range weights {
		weights[i] = 1.0 / float64(n)
	}
	
	return &BLEUMetric{
		BaseMetric: BaseMetric{name: "bleu"},
		n:          n,
		weights:    weights,
		smoothing:  true,
	}
}

// Compute implements Metric.Compute.
func (m *BLEUMetric) Compute(prediction string, reference string) float64 {
	predTokens := strings.Fields(strings.ToLower(prediction))
	refTokens := strings.Fields(strings.ToLower(reference))
	
	return m.computeScore(predTokens, refTokens)
}

// ComputeBatch implements Metric.ComputeBatch.
func (m *BLEUMetric) ComputeBatch(predictions, references []string) []float64 {
	return m.BaseMetric.ComputeBatch(predictions, references, m.Compute)
}

// computeScore calculates BLEU score for tokenized text.
func (m *BLEUMetric) computeScore(predTokens, refTokens []string) float64 {
	if len(predTokens) == 0 {
		return 0.0
	}
	
	// Calculate n-gram precisions
	logPrecisionSum := 0.0
	
	for n := 1; n <= m.n; n++ {
		predNgrams := m.getNgrams(predTokens, n)
		refNgrams := m.getNgrams(refTokens, n)
		
		if len(predNgrams) == 0 {
			// Apply smoothing for zero counts
			if m.smoothing {
				logPrecisionSum += m.weights[n-1] * math.Log(1.0/float64(len(predTokens)))
			} else {
				return 0.0
			}
			continue
		}
		
		// Count clipped matches
		matches := 0
		for ngram, predCount := range predNgrams {
			refCount := refNgrams[ngram]
			matches += min(predCount, refCount)
		}
		
		precision := float64(matches) / float64(len(predNgrams))
		
		// Apply smoothing for zero precision
		if precision == 0.0 {
			if m.smoothing {
				precision = 1.0 / float64(2*len(predNgrams))
			} else {
				return 0.0
			}
		}
		
		logPrecisionSum += m.weights[n-1] * math.Log(precision)
	}
	
	// Brevity penalty
	bp := m.brevityPenalty(len(predTokens), len(refTokens))
	
	// BLEU score
	score := bp * math.Exp(logPrecisionSum)
	
	return score
}

// getNgrams extracts n-grams from tokens.
func (m *BLEUMetric) getNgrams(tokens []string, n int) map[string]int {
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

// brevityPenalty calculates the brevity penalty.
func (m *BLEUMetric) brevityPenalty(predLen, refLen int) float64 {
	if predLen == 0 {
		return 0.0
	}
	
	if predLen > refLen {
		return 1.0
	}
	
	return math.Exp(1.0 - float64(refLen)/float64(predLen))
}

// CorpusBLEU computes corpus-level BLEU score.
func (m *BLEUMetric) CorpusBLEU(predictions, references []string) float64 {
	if len(predictions) != len(references) || len(predictions) == 0 {
		return 0.0
	}
	
	totalPredLen := 0
	totalRefLen := 0
	ngramMatches := make([]int, m.n)
	ngramCounts := make([]int, m.n)
	
	for i := range predictions {
		predTokens := strings.Fields(strings.ToLower(predictions[i]))
		refTokens := strings.Fields(strings.ToLower(references[i]))
		
		totalPredLen += len(predTokens)
		totalRefLen += len(refTokens)
		
		// Accumulate n-gram statistics
		for n := 1; n <= m.n; n++ {
			predNgrams := m.getNgrams(predTokens, n)
			refNgrams := m.getNgrams(refTokens, n)
			
			matches := 0
			for ngram, predCount := range predNgrams {
				refCount := refNgrams[ngram]
				matches += min(predCount, refCount)
			}
			
			ngramMatches[n-1] += matches
			ngramCounts[n-1] += len(predNgrams)
		}
	}
	
	// Calculate log precision sum
	logPrecisionSum := 0.0
	for n := 0; n < m.n; n++ {
		if ngramCounts[n] == 0 {
			if m.smoothing {
				logPrecisionSum += m.weights[n] * math.Log(1.0/float64(totalPredLen))
			} else {
				return 0.0
			}
			continue
		}
		
		precision := float64(ngramMatches[n]) / float64(ngramCounts[n])
		if precision == 0.0 {
			if m.smoothing {
				precision = 1.0 / float64(2*ngramCounts[n])
			} else {
				return 0.0
			}
		}
		
		logPrecisionSum += m.weights[n] * math.Log(precision)
	}
	
	// Brevity penalty
	bp := m.brevityPenalty(totalPredLen, totalRefLen)
	
	// Corpus BLEU score
	score := bp * math.Exp(logPrecisionSum)
	
	return score
}
