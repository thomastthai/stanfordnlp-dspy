# DSPy Go Teleprompt Optimizers

This package provides a comprehensive set of optimizers for DSPy modules in Go, achieving feature parity with Python DSPy.

## Overview

Teleprompt optimizers improve module performance by automatically tuning prompts, demonstrations, and other parameters using training data and evaluation metrics.

## Available Optimizers

### 1. BootstrapTrace

Trace-based bootstrapping optimizer that captures execution paths during prediction and bootstraps based on successful traces.

**Use Case**: When you need detailed execution traces to understand and improve module behavior.

**Example**:
```go
optimizer := teleprompt.NewBootstrapTrace(maxDemos).
    WithTraceMode("full").
    WithMaxLabeledDemos(20).
    WithNumThreads(4)

optimizedModule, err := optimizer.Compile(ctx, module, trainset, metric)
```

**Parameters**:
- `MaxBootstrappedDemos`: Maximum number of demonstrations to bootstrap
- `MaxLabeledDemos`: Maximum number of labeled examples to use
- `TraceMode`: Level of trace capture ("full", "minimal", "selective")
- `Teacher`: Optional teacher module for generating traces
- `NumThreads`: Number of threads for parallel evaluation

### 2. COPRO (Coordinate Ascent Prompt Optimization)

Optimizes prompts using coordinate ascent, iteratively refining instructions for each predictor in the module.

**Use Case**: When you need to optimize prompt instructions systematically.

**Example**:
```go
optimizer := teleprompt.NewCOPRO(metric).
    WithBreadth(10).
    WithDepth(3).
    WithInitTemperature(1.4)

optimizedModule, err := optimizer.Compile(ctx, module, trainset, metric)
```

**Parameters**:
- `Breadth`: Number of prompt candidates per iteration
- `Depth`: Number of optimization rounds
- `InitTemperature`: Temperature for prompt generation
- `PromptModel`: Model used for generating prompt variations

### 3. Ensemble

Combines multiple optimized modules using voting or averaging strategies.

**Use Case**: When you want to improve robustness by combining multiple models.

**Example**:
```go
optimizer := teleprompt.NewEnsemble().
    WithSize(5).
    WithReduceFn(customReduceFn)

optimizedModule, err := optimizer.Compile(ctx, module, trainset, metric)
```

**Parameters**:
- `Size`: Number of programs to include in ensemble
- `ReduceFn`: Function to aggregate predictions
- `Deterministic`: Whether sampling is deterministic

### 4. GRPO (Group Relative Policy Optimization)

Reinforcement learning-based optimizer using policy gradients with group-relative reward normalization.

**Use Case**: For complex optimization tasks requiring RL-style training.

**Example**:
```go
optimizer := teleprompt.NewGRPO(metric).
    WithNumEpochs(3).
    WithBatchSize(32).
    WithLearningRate(3e-4).
    WithGamma(0.99).
    WithClipRange(0.2)

optimizedModule, err := optimizer.Compile(ctx, module, trainset, metric)
```

**Parameters**:
- `NumEpochs`: Number of training epochs
- `BatchSize`: Batch size for training
- `LearningRate`: Learning rate for optimization
- `Gamma`: Discount factor for rewards
- `LambdaValue`: GAE lambda parameter
- `ClipRange`: PPO clip range

### 5. InferRules

Automatically infers natural language rules from training examples and uses them to improve instructions.

**Use Case**: When you want to extract and apply patterns from successful examples.

**Example**:
```go
optimizer := teleprompt.NewInferRules(metric).
    WithNumCandidates(10).
    WithNumRules(5)

optimizedModule, err := optimizer.Compile(ctx, module, trainset, metric)
```

**Parameters**:
- `NumCandidates`: Number of candidate programs to generate
- `NumRules`: Number of rules to infer per predictor
- `MaxErrors`: Maximum errors allowed during evaluation

### 6. KNNFewShot

Uses k-nearest neighbors to dynamically select relevant demonstrations at runtime based on input similarity.

**Use Case**: When different inputs require different examples for optimal performance.

**Example**:
```go
optimizer := teleprompt.NewKNNFewShot(k, trainset, vectorizer).
    WithFewShotBootstrapArgs(args)

optimizedModule, err := optimizer.Compile(ctx, module, trainset, metric)
```

**Parameters**:
- `K`: Number of nearest neighbors to retrieve
- `Trainset`: Set of examples to search
- `Vectorizer`: Function to convert examples to vectors
- `Metric`: Distance metric ("cosine", "euclidean", "manhattan")

### 7. SignatureOptimizer

Optimizes signature field descriptions, instructions, and field ordering. **Note**: Deprecated in favor of COPRO.

**Use Case**: Legacy support for signature optimization (use COPRO for new code).

**Example**:
```go
optimizer := teleprompt.NewSignatureOptimizer(metric).
    WithOptimizeInstructions(true).
    WithOptimizeDescriptions(true).
    WithNumCandidates(10)

optimizedModule, err := optimizer.Compile(ctx, module, trainset, metric)
```

### 8. SIMBA

Sampling-based optimization using bandit algorithms (UCB, Thompson Sampling, Epsilon-Greedy) with importance weighting.

**Use Case**: When you need efficient exploration-exploitation trade-offs during optimization.

**Example**:
```go
optimizer := teleprompt.NewSIMBA(metric).
    WithBanditAlgorithm("ucb").
    WithNumCandidates(20).
    WithNumIterations(10)

optimizedModule, err := optimizer.Compile(ctx, module, trainset, metric)
```

**Parameters**:
- `BanditAlgorithm`: Algorithm to use ("ucb", "thompson", "epsilon_greedy")
- `NumCandidates`: Number of candidate programs
- `NumIterations`: Number of optimization iterations
- `ExplorationParameter`: Exploration vs exploitation trade-off

## SIMBA Utility Functions

The package provides standalone utility functions for bandit algorithms and reinforcement learning:

### Bandit Algorithms

```go
// Upper Confidence Bound
armIdx := teleprompt.UCBSelection(armCounts, armMeans, totalCount, explorationParam)

// Thompson Sampling
armIdx := teleprompt.ThompsonSampling(armSuccesses, armFailures)

// Epsilon-Greedy
armIdx := teleprompt.EpsilonGreedy(armMeans, epsilon)

// Softmax Selection
armIdx := teleprompt.SoftmaxSelection(armMeans, temperature)
```

### Importance Sampling

```go
weight := teleprompt.ImportanceWeight(probability, samplingProbability)
```

### Reward Processing

```go
// Normalize rewards to mean=0, std=1
normalized := teleprompt.NormalizeRewards(rewards)

// Compute entropy bonus
entropy := teleprompt.ComputeEntropyBonus(probabilities)
```

### Reinforcement Learning Utilities

```go
// Generalized Advantage Estimation
advantages := teleprompt.ComputeGAE(rewards, values, gamma, lambda)

// Gradient Clipping
clipped := teleprompt.ClipGradients(gradients, maxNorm)
```

## Choosing an Optimizer

| Optimizer | Best For | Complexity | Speed |
|-----------|----------|------------|-------|
| LabeledFewShot | Simple few-shot learning | Low | Fast |
| BootstrapTrace | Understanding execution paths | Medium | Medium |
| COPRO | Systematic prompt optimization | Medium | Medium |
| InferRules | Rule extraction from examples | Medium | Medium |
| KNNFewShot | Dynamic example selection | Medium | Fast |
| SIMBA | Efficient exploration | Medium | Medium |
| Ensemble | Robustness via voting | Medium | Slow |
| GRPO | Complex RL-style optimization | High | Slow |

## Common Patterns

### Basic Optimization

```go
import (
    "context"
    "github.com/stanfordnlp/dspy/internal/teleprompt"
    "github.com/stanfordnlp/dspy/internal/primitives"
)

// Create optimizer
optimizer := teleprompt.NewBootstrapTrace(10)

// Define metric
metric := func(example *primitives.Example, prediction *primitives.Prediction) float64 {
    // Your evaluation logic
    return score
}

// Compile module
ctx := context.Background()
optimizedModule, err := optimizer.Compile(ctx, module, trainset, metric)
```

### Multi-Stage Optimization

```go
// Stage 1: Bootstrap with few-shot examples
stage1 := teleprompt.NewBootstrapFewShot(16)
module1, _ := stage1.Compile(ctx, module, trainset, metric)

// Stage 2: Refine with COPRO
stage2 := teleprompt.NewCOPRO(metric).WithDepth(3)
module2, _ := stage2.Compile(ctx, module1, trainset, metric)

// Stage 3: Create ensemble
stage3 := teleprompt.NewEnsemble().WithSize(5)
finalModule, _ := stage3.Compile(ctx, module2, trainset, metric)
```

### Custom Metric

```go
// Metric that returns a score between 0 and 1
customMetric := func(example *primitives.Example, prediction *primitives.Prediction) float64 {
    expected := example.Labels()["answer"].(string)
    actual := prediction.Get("answer").(string)
    
    if expected == actual {
        return 1.0
    }
    return 0.0
}

optimizer := teleprompt.NewCOPRO(customMetric)
```

## Testing

Run tests for all optimizers:

```bash
go test ./internal/teleprompt/... -v
```

Run specific optimizer tests:

```bash
go test ./internal/teleprompt/... -v -run TestBootstrapTrace
go test ./internal/teleprompt/... -v -run TestGRPO
go test ./internal/teleprompt/... -v -run TestSimbaUtils
```

## Implementation Notes

### Placeholder Implementations

Some optimizers have placeholder implementations for complex functionality:

- **GRPO**: Policy gradient updates and value function estimation
- **KNNFewShot**: Embedding generation and vector similarity search
- **Ensemble**: Multiple program management and reduction functions
- **BootstrapTrace**: Trace capture mechanism

These can be extended based on specific requirements.

### Thread Safety

All optimizers are designed to be used in concurrent contexts and support cancellation via `context.Context`.

### Error Handling

Optimizers return errors for:
- Empty training sets
- Missing required metrics
- Invalid parameters
- Compilation failures

Always check errors returned by `Compile()`.

## References

- [DSPy Python Documentation](https://github.com/stanfordnlp/dspy)
- Python teleprompt implementations in `dspy/teleprompt/`

## Contributing

When adding new optimizers:

1. Implement the `Teleprompt` interface
2. Extend `BaseTeleprompt` for common functionality
3. Add comprehensive tests in `*_test.go`
4. Update this documentation
5. Add usage examples

## License

Same as the main DSPy project.
