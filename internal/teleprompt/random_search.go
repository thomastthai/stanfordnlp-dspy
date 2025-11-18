package teleprompt

import (
	"context"
	"fmt"
	"math/rand"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// RandomSearch performs random search over hyperparameters with BootstrapFewShot.
// Based on dspy/teleprompt/random_search.py (BootstrapFewShotWithRandomSearch)
type RandomSearch struct {
	*BaseTeleprompt

	// Metric to optimize
	Metric interface{}

	// TeacherSettings for bootstrapping
	TeacherSettings map[string]interface{}

	// MaxBootstrappedDemos is the maximum number of bootstrapped demos
	MaxBootstrappedDemos int

	// MaxLabeledDemos is the maximum number of labeled demos
	MaxLabeledDemos int

	// MaxRounds is the maximum number of bootstrapping rounds
	MaxRounds int

	// NumCandidatePrograms is the number of random configurations to try
	NumCandidatePrograms int

	// NumThreads for parallel evaluation
	NumThreads int

	// MaxErrors allowed during evaluation
	MaxErrors int

	// StopAtScore stops early if this score is reached
	StopAtScore *float64

	// MetricThreshold for filtering bootstrapped examples
	MetricThreshold *float64

	// MinNumSamples is the minimum number of demos to sample
	MinNumSamples int

	// LabeledSample determines if labeled demos should be sampled
	LabeledSample bool

	// Seed for random number generation
	Seed int64
}

// NewRandomSearch creates a new RandomSearch optimizer.
func NewRandomSearch(metric interface{}) *RandomSearch {
	return &RandomSearch{
		BaseTeleprompt:       NewBaseTeleprompt("RandomSearch"),
		Metric:               metric,
		TeacherSettings:      make(map[string]interface{}),
		MaxBootstrappedDemos: 4,
		MaxLabeledDemos:      16,
		MaxRounds:            1,
		NumCandidatePrograms: 16,
		NumThreads:           1,
		MaxErrors:            0,
		MinNumSamples:        1,
		LabeledSample:        true,
		Seed:                 0,
	}
}

// WithTeacherSettings sets the teacher settings.
func (r *RandomSearch) WithTeacherSettings(settings map[string]interface{}) *RandomSearch {
	r.TeacherSettings = settings
	return r
}

// WithMaxBootstrappedDemos sets the max bootstrapped demos.
func (r *RandomSearch) WithMaxBootstrappedDemos(max int) *RandomSearch {
	r.MaxBootstrappedDemos = max
	return r
}

// WithMaxLabeledDemos sets the max labeled demos.
func (r *RandomSearch) WithMaxLabeledDemos(max int) *RandomSearch {
	r.MaxLabeledDemos = max
	return r
}

// WithNumCandidatePrograms sets the number of candidates to try.
func (r *RandomSearch) WithNumCandidatePrograms(num int) *RandomSearch {
	r.NumCandidatePrograms = num
	return r
}

// WithStopAtScore sets the early stopping score.
func (r *RandomSearch) WithStopAtScore(score float64) *RandomSearch {
	r.StopAtScore = &score
	return r
}

// Compile implements Teleprompt.Compile.
// It tries multiple random configurations and returns the best one.
func (r *RandomSearch) Compile(ctx context.Context, module primitives.Module, trainset []*primitives.Example, metric interface{}) (primitives.Module, error) {
	if metric != nil {
		r.Metric = metric
	}

	if r.Metric == nil {
		return nil, fmt.Errorf("metric is required for RandomSearch")
	}

	// Use provided trainset, or split if valset is needed
	valset := trainset // For simplicity, use same set for validation

	type candidateResult struct {
		program   primitives.Module
		score     float64
		subscores []float64
		seed      int
	}

	var bestProgram primitives.Module
	var bestScore float64
	scores := make([]float64, 0)
	candidates := make([]candidateResult, 0)

	// Try different random configurations
	// seed -3: zero-shot
	// seed -2: labels only
	// seed -1: unshuffled few-shot
	// seed >= 0: shuffled few-shot with random size
	for seed := -3; seed < r.NumCandidatePrograms; seed++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		var program primitives.Module
		var err error

		trainsetCopy := make([]*primitives.Example, len(trainset))
		copy(trainsetCopy, trainset)

		if seed == -3 {
			// Zero-shot: reset copy without demos
			program = module.Copy()
		} else if seed == -2 {
			// Labels only: use LabeledFewShot
			labeledOptimizer := NewLabeledFewShot(r.MaxLabeledDemos).WithSample(r.LabeledSample)
			program, err = labeledOptimizer.Compile(ctx, module, trainsetCopy, r.Metric)
			if err != nil {
				return nil, fmt.Errorf("labeled few-shot failed: %w", err)
			}
		} else if seed == -1 {
			// Unshuffled BootstrapFewShot
			bootstrap := NewBootstrapFewShot(r.MaxBootstrappedDemos).
				WithMaxLabeledDemos(r.MaxLabeledDemos).
				WithMaxRounds(r.MaxRounds)
			program, err = bootstrap.Compile(ctx, module, trainsetCopy, r.Metric)
			if err != nil {
				return nil, fmt.Errorf("bootstrap few-shot failed: %w", err)
			}
		} else {
			// Random shuffled BootstrapFewShot with random size
			seedRng := rand.New(rand.NewSource(int64(seed)))
			seedRng.Shuffle(len(trainsetCopy), func(i, j int) {
				trainsetCopy[i], trainsetCopy[j] = trainsetCopy[j], trainsetCopy[i]
			})

			size := seedRng.Intn(r.MaxBootstrappedDemos-r.MinNumSamples+1) + r.MinNumSamples

			bootstrap := NewBootstrapFewShot(size).
				WithMaxLabeledDemos(r.MaxLabeledDemos).
				WithMaxRounds(r.MaxRounds)
			program, err = bootstrap.Compile(ctx, module, trainsetCopy, r.Metric)
			if err != nil {
				return nil, fmt.Errorf("bootstrap few-shot (seed=%d) failed: %w", seed, err)
			}
		}

		// Evaluate the program on valset
		score, subscores, err := r.evaluateProgram(ctx, program, valset)
		if err != nil {
			return nil, fmt.Errorf("evaluation failed for seed %d: %w", seed, err)
		}

		scores = append(scores, score)
		candidates = append(candidates, candidateResult{
			program:   program,
			score:     score,
			subscores: subscores,
			seed:      seed,
		})

		// Track best program
		if bestProgram == nil || score > bestScore {
			bestProgram = program
			bestScore = score
		}

		// Check for early stopping
		if r.StopAtScore != nil && score >= *r.StopAtScore {
			break
		}
	}

	if bestProgram == nil {
		return nil, fmt.Errorf("no valid program found")
	}

	return bestProgram, nil
}

// evaluateProgram evaluates a program on a dataset.
func (r *RandomSearch) evaluateProgram(ctx context.Context, program primitives.Module, dataset []*primitives.Example) (float64, []float64, error) {
	// Simplified evaluation - in a full implementation, this would:
	// 1. Run the program on each example in the dataset
	// 2. Compute the metric for each prediction
	// 3. Return average score and individual subscores

	// For now, return a placeholder score
	subscores := make([]float64, len(dataset))
	for i := range subscores {
		subscores[i] = 0.5 // Placeholder
	}

	var totalScore float64
	for _, s := range subscores {
		totalScore += s
	}
	avgScore := totalScore / float64(len(subscores))

	return avgScore, subscores, nil
}
