// Package predict provides prediction modules that interact with language models.
package predict

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// KNN implements k-nearest neighbor demo selection.
type KNN struct {
	*primitives.BaseModule

	// K is the number of nearest neighbors to retrieve
	K int

	// Trainset contains training examples
	Trainset []*primitives.Example

	// Vectorizer embeds text into vectors
	Vectorizer func(string) ([]float64, error)

	// TrainsetVectors are pre-computed vectors for the training set
	TrainsetVectors [][]float64

	// DistanceMetric determines how to calculate distance
	// Supported: "cosine", "euclidean"
	DistanceMetric string

	// Config contains additional configuration
	Config map[string]interface{}
}

// NewKNN creates a new KNN module for demo selection.
func NewKNN(k int, trainset []*primitives.Example, vectorizer func(string) ([]float64, error)) (*KNN, error) {
	if k <= 0 {
		k = 3 // Default to 3 neighbors
	}

	knn := &KNN{
		BaseModule:     primitives.NewBaseModule(),
		K:              k,
		Trainset:       trainset,
		Vectorizer:     vectorizer,
		DistanceMetric: "cosine",
		Config:         make(map[string]interface{}),
	}

	// Pre-compute vectors for trainset
	if err := knn.computeTrainsetVectors(); err != nil {
		return nil, fmt.Errorf("failed to compute trainset vectors: %w", err)
	}

	return knn, nil
}

// computeTrainsetVectors pre-computes embeddings for all training examples.
func (k *KNN) computeTrainsetVectors() error {
	k.TrainsetVectors = make([][]float64, len(k.Trainset))

	for i, example := range k.Trainset {
		// Concatenate input fields into a single string
		text := k.exampleToText(example)

		// Compute embedding
		vector, err := k.Vectorizer(text)
		if err != nil {
			return fmt.Errorf("failed to vectorize example %d: %w", i, err)
		}

		k.TrainsetVectors[i] = vector
	}

	return nil
}

// exampleToText converts an example to a text string for embedding.
func (k *KNN) exampleToText(example *primitives.Example) string {
	// Get inputs from example
	inputs := example.Inputs()
	text := ""
	for key, value := range inputs {
		if text != "" {
			text += " | "
		}
		text += fmt.Sprintf("%s: %v", key, value)
	}
	return text
}

// Forward finds k-nearest neighbors for the given inputs.
func (k *KNN) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Convert inputs to text
	text := ""
	for key, value := range inputs {
		if text != "" {
			text += " | "
		}
		text += fmt.Sprintf("%s: %v", key, value)
	}

	// Compute embedding for input
	queryVector, err := k.Vectorizer(text)
	if err != nil {
		return nil, fmt.Errorf("failed to vectorize query: %w", err)
	}

	// Find k-nearest neighbors
	neighbors := k.findKNearest(queryVector)

	// Create output
	output := map[string]interface{}{
		"neighbors": neighbors,
		"k":         k.K,
	}

	pred := primitives.NewPrediction(output)
	pred.SetMetadata("knn_retrieval", true)
	pred.SetMetadata("distance_metric", k.DistanceMetric)

	return pred, nil
}

// findKNearest finds the k-nearest neighbors based on distance.
func (k *KNN) findKNearest(queryVector []float64) []*primitives.Example {
	type scoredExample struct {
		example  *primitives.Example
		distance float64
	}

	// Calculate distances to all training examples
	scored := make([]scoredExample, len(k.Trainset))
	for i, trainVector := range k.TrainsetVectors {
		distance := k.calculateDistance(queryVector, trainVector)
		scored[i] = scoredExample{
			example:  k.Trainset[i],
			distance: distance,
		}
	}

	// Sort by distance (ascending for euclidean, descending for cosine similarity)
	if k.DistanceMetric == "cosine" {
		// For cosine similarity, higher is better (more similar)
		sort.Slice(scored, func(i, j int) bool {
			return scored[i].distance > scored[j].distance
		})
	} else {
		// For euclidean, lower is better (closer)
		sort.Slice(scored, func(i, j int) bool {
			return scored[i].distance < scored[j].distance
		})
	}

	// Return top k
	neighbors := make([]*primitives.Example, 0, k.K)
	for i := 0; i < k.K && i < len(scored); i++ {
		neighbors = append(neighbors, scored[i].example)
	}

	return neighbors
}

// calculateDistance computes distance between two vectors.
func (k *KNN) calculateDistance(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		return math.Inf(1) // Return infinity for mismatched dimensions
	}

	switch k.DistanceMetric {
	case "cosine":
		return cosineSimilarity(v1, v2)
	case "euclidean":
		return euclideanDistance(v1, v2)
	default:
		return euclideanDistance(v1, v2)
	}
}

// cosineSimilarity calculates cosine similarity between two vectors.
func cosineSimilarity(v1, v2 []float64) float64 {
	var dotProduct, norm1, norm2 float64

	for i := range v1 {
		dotProduct += v1[i] * v2[i]
		norm1 += v1[i] * v1[i]
		norm2 += v2[i] * v2[i]
	}

	if norm1 == 0 || norm2 == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2))
}

// euclideanDistance calculates Euclidean distance between two vectors.
func euclideanDistance(v1, v2 []float64) float64 {
	var sum float64
	for i := range v1 {
		diff := v1[i] - v2[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// Copy creates a deep copy of the KNN module.
func (k *KNN) Copy() primitives.Module {
	newKNN := &KNN{
		BaseModule:      primitives.NewBaseModule(),
		K:               k.K,
		Trainset:        k.Trainset, // Trainset is read-only, safe to share
		Vectorizer:      k.Vectorizer,
		TrainsetVectors: k.TrainsetVectors, // Pre-computed, safe to share
		DistanceMetric:  k.DistanceMetric,
		Config:          make(map[string]interface{}),
	}

	for key, value := range k.Config {
		newKNN.Config[key] = value
	}

	return newKNN
}

// NamedParameters returns all parameters in this module.
func (k *KNN) NamedParameters() []primitives.NamedParameter {
	return []primitives.NamedParameter{}
}
