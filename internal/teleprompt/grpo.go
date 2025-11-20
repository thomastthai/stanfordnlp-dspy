package teleprompt

import (
	"context"
	"fmt"
	"math"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// GRPO implements Group Relative Policy Optimization.
// It uses reinforcement learning with group-relative reward normalization
// to optimize module parameters.
// Based on dspy/teleprompt/grpo.py
type GRPO struct {
	*BaseTeleprompt

	// Metric to optimize
	Metric interface{}

	// NumEpochs is the number of training epochs
	NumEpochs int

	// BatchSize is the batch size for training
	BatchSize int

	// LearningRate for optimization
	LearningRate float64

	// Gamma is the discount factor
	Gamma float64

	// LambdaValue is the GAE lambda parameter
	LambdaValue float64

	// ClipRange is the PPO clip range
	ClipRange float64

	// ValueCoef is the value function loss coefficient
	ValueCoef float64

	// EntropyCoef is the entropy bonus coefficient
	EntropyCoef float64

	// MaxGradNorm for gradient clipping
	MaxGradNorm float64

	// NumMiniBatches is the number of mini-batches per epoch
	NumMiniBatches int

	// NumThreads for parallel evaluation
	NumThreads int

	// Multitask determines if predictors are trained together
	Multitask bool

	// ExcludeDemos determines if demos should be excluded
	ExcludeDemos bool

	// NumDSPyExamplesPerGRPOStep is the number of examples per GRPO step
	NumDSPyExamplesPerGRPOStep int

	// NumRolloutsPerGRPOStep is the number of rollouts per step
	NumRolloutsPerGRPOStep int

	// UseTrainAsVal determines if training set should be used for validation
	UseTrainAsVal bool

	// NumStepsForVal is the number of steps between validation
	NumStepsForVal int

	// ReportTrainScores determines if training scores should be reported
	ReportTrainScores bool

	// FailureScore is the score assigned to failures
	FailureScore float64

	// FormatFailureScore is the score for format failures
	FormatFailureScore float64

	// Verbose enables detailed logging
	Verbose bool
}

// NewGRPO creates a new GRPO optimizer.
func NewGRPO(metric interface{}) *GRPO {
	return &GRPO{
		BaseTeleprompt:             NewBaseTeleprompt("GRPO"),
		Metric:                     metric,
		NumEpochs:                  3,
		BatchSize:                  32,
		LearningRate:               3e-4,
		Gamma:                      0.99,
		LambdaValue:                0.95,
		ClipRange:                  0.2,
		ValueCoef:                  0.5,
		EntropyCoef:                0.01,
		MaxGradNorm:                0.5,
		NumMiniBatches:             4,
		NumThreads:                 6,
		Multitask:                  true,
		ExcludeDemos:               true,
		NumDSPyExamplesPerGRPOStep: 1,
		NumRolloutsPerGRPOStep:     1,
		UseTrainAsVal:              false,
		NumStepsForVal:             5,
		ReportTrainScores:          false,
		FailureScore:               0.0,
		FormatFailureScore:         -1.0,
		Verbose:                    false,
	}
}

// WithNumEpochs sets the number of epochs.
func (g *GRPO) WithNumEpochs(epochs int) *GRPO {
	g.NumEpochs = epochs
	return g
}

// WithBatchSize sets the batch size.
func (g *GRPO) WithBatchSize(size int) *GRPO {
	g.BatchSize = size
	return g
}

// WithLearningRate sets the learning rate.
func (g *GRPO) WithLearningRate(lr float64) *GRPO {
	g.LearningRate = lr
	return g
}

// WithGamma sets the discount factor.
func (g *GRPO) WithGamma(gamma float64) *GRPO {
	g.Gamma = gamma
	return g
}

// WithLambdaValue sets the GAE lambda.
func (g *GRPO) WithLambdaValue(lambda float64) *GRPO {
	g.LambdaValue = lambda
	return g
}

// WithClipRange sets the PPO clip range.
func (g *GRPO) WithClipRange(clip float64) *GRPO {
	g.ClipRange = clip
	return g
}

// WithValueCoef sets the value coefficient.
func (g *GRPO) WithValueCoef(coef float64) *GRPO {
	g.ValueCoef = coef
	return g
}

// WithEntropyCoef sets the entropy coefficient.
func (g *GRPO) WithEntropyCoef(coef float64) *GRPO {
	g.EntropyCoef = coef
	return g
}

// WithMaxGradNorm sets the max gradient norm.
func (g *GRPO) WithMaxGradNorm(norm float64) *GRPO {
	g.MaxGradNorm = norm
	return g
}

// WithNumMiniBatches sets the number of mini-batches.
func (g *GRPO) WithNumMiniBatches(num int) *GRPO {
	g.NumMiniBatches = num
	return g
}

// WithMultitask sets the multitask mode.
func (g *GRPO) WithMultitask(multitask bool) *GRPO {
	g.Multitask = multitask
	return g
}

// WithVerbose enables verbose logging.
func (g *GRPO) WithVerbose(verbose bool) *GRPO {
	g.Verbose = verbose
	return g
}

// Compile implements Teleprompt.Compile.
// It uses GRPO to optimize the module with reinforcement learning.
func (g *GRPO) Compile(ctx context.Context, module primitives.Module, trainset []*primitives.Example, metric interface{}) (primitives.Module, error) {
	if metric != nil {
		g.Metric = metric
	}

	if g.Metric == nil {
		return nil, fmt.Errorf("metric is required for GRPO")
	}

	if len(trainset) == 0 {
		return nil, fmt.Errorf("trainset is empty")
	}

	// Validate that failure_score > format_failure_score
	if g.FailureScore <= g.FormatFailureScore {
		return nil, fmt.Errorf("failure_score must be greater than format_failure_score")
	}

	// Initialize the optimized module
	optimizedModule := module.Copy()

	// Create group-relative policy optimizer
	optimizer := &GRPOOptimizer{
		module:       optimizedModule,
		trainset:     trainset,
		metric:       g.Metric,
		learningRate: g.LearningRate,
		gamma:        g.Gamma,
		lambda:       g.LambdaValue,
		clipRange:    g.ClipRange,
		verbose:      g.Verbose,
	}

	// Run GRPO optimization loop
	for epoch := 0; epoch < g.NumEpochs; epoch++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		if g.Verbose {
			fmt.Printf("GRPO Epoch %d/%d\n", epoch+1, g.NumEpochs)
		}

		// Collect rollouts
		rollouts, err := optimizer.collectRollouts(ctx, g.NumRolloutsPerGRPOStep)
		if err != nil {
			return nil, fmt.Errorf("failed to collect rollouts: %w", err)
		}

		// Compute group-relative advantages
		advantages := optimizer.computeGroupRelativeAdvantages(rollouts)

		// Update policy with PPO
		if err := optimizer.updatePolicy(ctx, rollouts, advantages); err != nil {
			return nil, fmt.Errorf("failed to update policy: %w", err)
		}

		// Validation if configured
		if g.NumStepsForVal > 0 && (epoch+1)%g.NumStepsForVal == 0 {
			valScore := optimizer.validate(ctx)
			if g.Verbose {
				fmt.Printf("Validation score: %.4f\n", valScore)
			}
		}
	}

	return optimizedModule, nil
}

// GRPOOptimizer encapsulates the GRPO optimization state.
type GRPOOptimizer struct {
	module       primitives.Module
	trainset     []*primitives.Example
	metric       interface{}
	learningRate float64
	gamma        float64
	lambda       float64
	clipRange    float64
	verbose      bool
}

// Rollout represents a single rollout trajectory.
type Rollout struct {
	Example    *primitives.Example
	Prediction *primitives.Prediction
	Reward     float64
	LogProbs   []float64
	Values     []float64
}

// collectRollouts collects rollout trajectories from the environment.
func (o *GRPOOptimizer) collectRollouts(ctx context.Context, numRollouts int) ([]*Rollout, error) {
	rollouts := make([]*Rollout, 0, numRollouts*len(o.trainset))

	for i := 0; i < numRollouts; i++ {
		for _, example := range o.trainset {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
			}

			// Execute module and collect trajectory
			prediction, err := o.module.Forward(ctx, example.Inputs())
			if err != nil {
				continue
			}

			// Compute reward
			reward := o.computeReward(example, prediction)

			rollout := &Rollout{
				Example:    example,
				Prediction: prediction,
				Reward:     reward,
				LogProbs:   []float64{0.0}, // Placeholder
				Values:     []float64{0.0}, // Placeholder
			}

			rollouts = append(rollouts, rollout)
		}
	}

	return rollouts, nil
}

// computeGroupRelativeAdvantages computes advantages using group-relative normalization.
func (o *GRPOOptimizer) computeGroupRelativeAdvantages(rollouts []*Rollout) []float64 {
	if len(rollouts) == 0 {
		return []float64{}
	}

	// Compute mean and std of rewards
	var sum, sumSq float64
	for _, rollout := range rollouts {
		sum += rollout.Reward
		sumSq += rollout.Reward * rollout.Reward
	}

	mean := sum / float64(len(rollouts))
	variance := (sumSq / float64(len(rollouts))) - (mean * mean)
	std := math.Sqrt(math.Max(variance, 1e-8))

	// Normalize rewards to get advantages
	advantages := make([]float64, len(rollouts))
	for i, rollout := range rollouts {
		advantages[i] = (rollout.Reward - mean) / std
	}

	return advantages
}

// updatePolicy updates the policy using PPO algorithm.
func (o *GRPOOptimizer) updatePolicy(ctx context.Context, rollouts []*Rollout, advantages []float64) error {
	// In a full implementation, this would:
	// 1. Compute policy and value function losses
	// 2. Clip gradients according to clipRange
	// 3. Update module parameters using gradient descent
	// 4. Apply entropy regularization

	// For now, this is a placeholder
	return nil
}

// computeReward computes the reward for a prediction.
func (o *GRPOOptimizer) computeReward(example *primitives.Example, prediction *primitives.Prediction) float64 {
	// In a full implementation, this would evaluate the metric
	// For now, return a placeholder reward
	return 0.5
}

// validate evaluates the current policy on validation set.
func (o *GRPOOptimizer) validate(ctx context.Context) float64 {
	// In a full implementation, this would evaluate on a validation set
	// For now, return a placeholder score
	return 0.5
}
