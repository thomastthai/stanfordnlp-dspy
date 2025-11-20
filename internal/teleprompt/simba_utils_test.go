package teleprompt

import (
	"math"
	"testing"
)

func TestUCBSelection(t *testing.T) {
	tests := []struct {
		name             string
		armCounts        []int
		armMeans         []float64
		totalCount       int
		explorationParam float64
		wantUnexplored   bool // Should select an unexplored arm
	}{
		{
			name:             "unexplored arm exists",
			armCounts:        []int{5, 0, 3},
			armMeans:         []float64{0.5, 0.0, 0.4},
			totalCount:       8,
			explorationParam: 2.0,
			wantUnexplored:   true,
		},
		{
			name:             "all arms explored",
			armCounts:        []int{5, 3, 2},
			armMeans:         []float64{0.5, 0.6, 0.4},
			totalCount:       10,
			explorationParam: 2.0,
			wantUnexplored:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			arm := UCBSelection(tt.armCounts, tt.armMeans, tt.totalCount, tt.explorationParam)

			if arm < 0 || arm >= len(tt.armCounts) {
				t.Errorf("UCBSelection returned invalid arm index: %d", arm)
			}

			if tt.wantUnexplored {
				// Should select the unexplored arm (index 1 in first test)
				if tt.armCounts[arm] != 0 {
					t.Errorf("Expected unexplored arm to be selected, got arm %d with count %d", arm, tt.armCounts[arm])
				}
			}
		})
	}
}

func TestThompsonSampling(t *testing.T) {
	tests := []struct {
		name         string
		armSuccesses []int
		armFailures  []int
	}{
		{
			name:         "basic sampling",
			armSuccesses: []int{10, 5, 8},
			armFailures:  []int{2, 5, 3},
		},
		{
			name:         "all zeros",
			armSuccesses: []int{0, 0, 0},
			armFailures:  []int{0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			arm := ThompsonSampling(tt.armSuccesses, tt.armFailures)

			if arm < 0 || arm >= len(tt.armSuccesses) {
				t.Errorf("ThompsonSampling returned invalid arm index: %d", arm)
			}
		})
	}
}

func TestEpsilonGreedy(t *testing.T) {
	tests := []struct {
		name     string
		armMeans []float64
		epsilon  float64
	}{
		{
			name:     "low epsilon (mostly exploit)",
			armMeans: []float64{0.5, 0.8, 0.3},
			epsilon:  0.1,
		},
		{
			name:     "high epsilon (mostly explore)",
			armMeans: []float64{0.5, 0.8, 0.3},
			epsilon:  0.9,
		},
		{
			name:     "zero epsilon (pure exploit)",
			armMeans: []float64{0.5, 0.8, 0.3},
			epsilon:  0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Run multiple times to test stochastic behavior
			selectedArms := make(map[int]int)

			for i := 0; i < 100; i++ {
				arm := EpsilonGreedy(tt.armMeans, tt.epsilon)

				if arm < 0 || arm >= len(tt.armMeans) {
					t.Errorf("EpsilonGreedy returned invalid arm index: %d", arm)
				}

				selectedArms[arm]++
			}

			// With epsilon=0, should always select best arm (index 1)
			if tt.epsilon == 0.0 {
				if len(selectedArms) != 1 || selectedArms[1] != 100 {
					t.Errorf("Expected pure exploitation to always select arm 1, got selections: %v", selectedArms)
				}
			}
		})
	}
}

func TestImportanceWeight(t *testing.T) {
	tests := []struct {
		name                string
		probability         float64
		samplingProbability float64
		wantWeight          float64
		wantPanic           bool
	}{
		{
			name:                "equal probabilities",
			probability:         0.5,
			samplingProbability: 0.5,
			wantWeight:          1.0,
		},
		{
			name:                "under-sampling",
			probability:         0.8,
			samplingProbability: 0.4,
			wantWeight:          2.0,
		},
		{
			name:                "over-sampling",
			probability:         0.2,
			samplingProbability: 0.8,
			wantWeight:          0.25,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			weight := ImportanceWeight(tt.probability, tt.samplingProbability)

			if math.Abs(weight-tt.wantWeight) > 1e-6 {
				t.Errorf("ImportanceWeight() = %v, want %v", weight, tt.wantWeight)
			}
		})
	}
}

func TestNormalizeRewards(t *testing.T) {
	tests := []struct {
		name    string
		rewards []float64
		wantLen int
	}{
		{
			name:    "empty rewards",
			rewards: []float64{},
			wantLen: 0,
		},
		{
			name:    "single reward",
			rewards: []float64{5.0},
			wantLen: 1,
		},
		{
			name:    "multiple rewards",
			rewards: []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			wantLen: 5,
		},
		{
			name:    "all same rewards",
			rewards: []float64{3.0, 3.0, 3.0, 3.0},
			wantLen: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			normalized := NormalizeRewards(tt.rewards)

			if len(normalized) != tt.wantLen {
				t.Errorf("NormalizeRewards() returned length %d, want %d", len(normalized), tt.wantLen)
			}

			if len(normalized) > 1 {
				// Check that mean is approximately 0
				sum := 0.0
				for _, r := range normalized {
					sum += r
				}
				mean := sum / float64(len(normalized))

				if math.Abs(mean) > 1e-6 {
					t.Errorf("Normalized rewards should have mean ~0, got %v", mean)
				}

				// Check that std is approximately 1 (unless all values were the same)
				allSame := true
				for i := 1; i < len(tt.rewards); i++ {
					if tt.rewards[i] != tt.rewards[0] {
						allSame = false
						break
					}
				}

				if !allSame {
					sumSq := 0.0
					for _, r := range normalized {
						sumSq += r * r
					}
					variance := sumSq / float64(len(normalized))
					std := math.Sqrt(variance)

					if math.Abs(std-1.0) > 0.1 {
						t.Errorf("Normalized rewards should have std ~1, got %v", std)
					}
				}
			}
		})
	}
}

func TestSoftmaxSelection(t *testing.T) {
	tests := []struct {
		name        string
		armMeans    []float64
		temperature float64
	}{
		{
			name:        "low temperature (more exploitation)",
			armMeans:    []float64{0.5, 0.8, 0.3},
			temperature: 0.1,
		},
		{
			name:        "high temperature (more exploration)",
			armMeans:    []float64{0.5, 0.8, 0.3},
			temperature: 2.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Run multiple times to test stochastic behavior
			selectedArms := make(map[int]int)

			for i := 0; i < 100; i++ {
				arm := SoftmaxSelection(tt.armMeans, tt.temperature)

				if arm < 0 || arm >= len(tt.armMeans) {
					t.Errorf("SoftmaxSelection returned invalid arm index: %d", arm)
				}

				selectedArms[arm]++
			}

			// With low temperature, should mostly select best arm (index 1)
			if tt.temperature < 0.5 {
				if selectedArms[1] < 50 { // At least 50% of the time
					t.Errorf("Expected low temperature to favor best arm, got selections: %v", selectedArms)
				}
			}
		})
	}
}

func TestComputeGAE(t *testing.T) {
	tests := []struct {
		name    string
		rewards []float64
		values  []float64
		gamma   float64
		lambda  float64
	}{
		{
			name:    "basic GAE",
			rewards: []float64{1.0, 2.0, 3.0},
			values:  []float64{0.5, 1.0, 1.5},
			gamma:   0.99,
			lambda:  0.95,
		},
		{
			name:    "empty inputs",
			rewards: []float64{},
			values:  []float64{},
			gamma:   0.99,
			lambda:  0.95,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			advantages := ComputeGAE(tt.rewards, tt.values, tt.gamma, tt.lambda)

			if len(advantages) != len(tt.rewards) {
				t.Errorf("ComputeGAE() returned length %d, want %d", len(advantages), len(tt.rewards))
			}
		})
	}
}

func TestClipGradients(t *testing.T) {
	tests := []struct {
		name      string
		gradients []float64
		maxNorm   float64
		wantClip  bool
	}{
		{
			name:      "no clipping needed",
			gradients: []float64{0.1, 0.2, 0.1},
			maxNorm:   1.0,
			wantClip:  false,
		},
		{
			name:      "clipping needed",
			gradients: []float64{5.0, 5.0, 5.0},
			maxNorm:   1.0,
			wantClip:  true,
		},
		{
			name:      "empty gradients",
			gradients: []float64{},
			maxNorm:   1.0,
			wantClip:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clipped := ClipGradients(tt.gradients, tt.maxNorm)

			if len(clipped) != len(tt.gradients) {
				t.Errorf("ClipGradients() returned length %d, want %d", len(clipped), len(tt.gradients))
			}

			if len(clipped) > 0 {
				// Compute norm of clipped gradients
				sumSq := 0.0
				for _, g := range clipped {
					sumSq += g * g
				}
				norm := math.Sqrt(sumSq)

				// Should be at most maxNorm
				if norm > tt.maxNorm+1e-6 {
					t.Errorf("Clipped gradients have norm %v, want at most %v", norm, tt.maxNorm)
				}
			}
		})
	}
}

func TestComputeEntropyBonus(t *testing.T) {
	tests := []struct {
		name          string
		probabilities []float64
		wantPositive  bool
	}{
		{
			name:          "uniform distribution (high entropy)",
			probabilities: []float64{0.25, 0.25, 0.25, 0.25},
			wantPositive:  true,
		},
		{
			name:          "deterministic (low entropy)",
			probabilities: []float64{1.0, 0.0, 0.0, 0.0},
			wantPositive:  false,
		},
		{
			name:          "empty probabilities",
			probabilities: []float64{},
			wantPositive:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entropy := ComputeEntropyBonus(tt.probabilities)

			if tt.wantPositive && entropy <= 0 {
				t.Errorf("Expected positive entropy, got %v", entropy)
			}

			if entropy < 0 {
				t.Errorf("Entropy should be non-negative, got %v", entropy)
			}
		})
	}
}
