package metrics

import (
	"context"
	"math"
	"strings"
)

// Embedder is an interface for generating embeddings.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// BERTScoreMetric computes BERTScore using contextual embeddings.
// Reference: Zhang et al. 2020
type BERTScoreMetric struct {
	BaseMetric
	embedder Embedder
	idfMode  bool
}

// NewBERTScore creates a new BERTScore metric.
func NewBERTScore(embedder Embedder) *BERTScoreMetric {
	return &BERTScoreMetric{
		BaseMetric: BaseMetric{name: "bertscore"},
		embedder:   embedder,
		idfMode:    false,
	}
}

// Compute implements Metric.Compute.
func (m *BERTScoreMetric) Compute(prediction string, reference string) float64 {
	scores, err := m.ComputeScores(context.Background(), prediction, reference)
	if err != nil {
		return 0.0
	}
	return scores.F1
}

// ComputeBatch implements Metric.ComputeBatch.
func (m *BERTScoreMetric) ComputeBatch(predictions, references []string) []float64 {
	scores := make([]float64, len(predictions))
	for i := range predictions {
		scores[i] = m.Compute(predictions[i], references[i])
	}
	return scores
}

// BERTScoreResult contains precision, recall, and F1 scores.
type BERTScoreResult struct {
	Precision float64
	Recall    float64
	F1        float64
}

// ComputeScores computes BERTScore with precision, recall, and F1.
func (m *BERTScoreMetric) ComputeScores(ctx context.Context, prediction string, reference string) (BERTScoreResult, error) {
	// Tokenize
	predTokens := strings.Fields(strings.ToLower(prediction))
	refTokens := strings.Fields(strings.ToLower(reference))
	
	if len(predTokens) == 0 || len(refTokens) == 0 {
		return BERTScoreResult{}, nil
	}
	
	// Get embeddings
	predEmbs, err := m.embedder.Embed(ctx, predTokens)
	if err != nil {
		return BERTScoreResult{}, err
	}
	
	refEmbs, err := m.embedder.Embed(ctx, refTokens)
	if err != nil {
		return BERTScoreResult{}, err
	}
	
	// Compute similarity matrix
	simMatrix := m.computeSimilarityMatrix(predEmbs, refEmbs)
	
	// Precision: for each prediction token, find max similarity with reference tokens
	precisionSum := 0.0
	for i := 0; i < len(predTokens); i++ {
		maxSim := 0.0
		for j := 0; j < len(refTokens); j++ {
			if simMatrix[i][j] > maxSim {
				maxSim = simMatrix[i][j]
			}
		}
		precisionSum += maxSim
	}
	precision := precisionSum / float64(len(predTokens))
	
	// Recall: for each reference token, find max similarity with prediction tokens
	recallSum := 0.0
	for j := 0; j < len(refTokens); j++ {
		maxSim := 0.0
		for i := 0; i < len(predTokens); i++ {
			if simMatrix[i][j] > maxSim {
				maxSim = simMatrix[i][j]
			}
		}
		recallSum += maxSim
	}
	recall := recallSum / float64(len(refTokens))
	
	// F1 score
	f1 := 0.0
	if precision+recall > 0 {
		f1 = 2 * precision * recall / (precision + recall)
	}
	
	return BERTScoreResult{
		Precision: precision,
		Recall:    recall,
		F1:        f1,
	}, nil
}

// computeSimilarityMatrix computes cosine similarity between embeddings.
func (m *BERTScoreMetric) computeSimilarityMatrix(embs1, embs2 [][]float32) [][]float64 {
	matrix := make([][]float64, len(embs1))
	for i := range matrix {
		matrix[i] = make([]float64, len(embs2))
		for j := range matrix[i] {
			matrix[i][j] = m.cosineSimilarity(embs1[i], embs2[j])
		}
	}
	return matrix
}

// cosineSimilarity computes cosine similarity between two vectors.
func (m *BERTScoreMetric) cosineSimilarity(v1, v2 []float32) float64 {
	if len(v1) != len(v2) {
		return 0.0
	}
	
	dotProduct := 0.0
	norm1 := 0.0
	norm2 := 0.0
	
	for i := 0; i < len(v1); i++ {
		dotProduct += float64(v1[i]) * float64(v2[i])
		norm1 += float64(v1[i]) * float64(v1[i])
		norm2 += float64(v2[i]) * float64(v2[i])
	}
	
	if norm1 == 0 || norm2 == 0 {
		return 0.0
	}
	
	return dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2))
}

// ComputeBatchScores computes BERTScore for multiple pairs.
func (m *BERTScoreMetric) ComputeBatchScores(ctx context.Context, predictions, references []string) ([]BERTScoreResult, error) {
	if len(predictions) != len(references) {
		return nil, nil
	}
	
	results := make([]BERTScoreResult, len(predictions))
	for i := range predictions {
		score, err := m.ComputeScores(ctx, predictions[i], references[i])
		if err != nil {
			return nil, err
		}
		results[i] = score
	}
	
	return results, nil
}
