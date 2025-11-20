// Package evaluate provides evaluation functionality for DSPy modules.
package evaluate

import (
	"context"
	"fmt"
	"sync"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/primitives"
)

// Metric is a function that evaluates a prediction against an example.
// Returns a score between 0 and 1, where 1 is perfect.
type Metric func(example *primitives.Example, prediction *primitives.Prediction) float64

// Evaluator evaluates a module on a dataset using a metric.
type Evaluator struct {
	metric          Metric
	numThreads      int
	displayProgress bool
	autoMetrics     []*AutoMetric
	useLMJudge      bool
	lmJudge         *LMJudge
}

// NewEvaluator creates a new evaluator.
func NewEvaluator(metric Metric) *Evaluator {
	return &Evaluator{
		metric:          metric,
		numThreads:      1,
		displayProgress: true,
	}
}

// WithNumThreads sets the number of parallel evaluation threads.
func (e *Evaluator) WithNumThreads(n int) *Evaluator {
	e.numThreads = n
	return e
}

// WithDisplayProgress sets whether to display progress.
func (e *Evaluator) WithDisplayProgress(display bool) *Evaluator {
	e.displayProgress = display
	return e
}

// EvaluationResult contains the results of an evaluation.
type EvaluationResult struct {
	// TotalScore is the sum of all scores
	TotalScore float64

	// Count is the number of examples evaluated
	Count int

	// AverageScore is the average score across all examples
	AverageScore float64

	// Scores contains individual scores for each example
	Scores []float64
}

// Evaluate runs the evaluation on a dataset.
func (e *Evaluator) Evaluate(ctx context.Context, module primitives.Module, dataset []*primitives.Example) (*EvaluationResult, error) {
	if len(dataset) == 0 {
		return nil, fmt.Errorf("dataset is empty")
	}

	result := &EvaluationResult{
		Scores: make([]float64, len(dataset)),
	}

	// Sequential evaluation for now
	// TODO: Implement parallel evaluation with numThreads
	for i, example := range dataset {
		// Forward pass
		prediction, err := module.Forward(ctx, example.ToMap())
		if err != nil {
			return nil, fmt.Errorf("forward pass failed on example %d: %w", i, err)
		}

		// Compute metric
		score := e.metric(example, prediction)
		result.Scores[i] = score
		result.TotalScore += score
		result.Count++

		if e.displayProgress && (i+1)%10 == 0 {
			fmt.Printf("Evaluated %d/%d examples\n", i+1, len(dataset))
		}
	}

	result.AverageScore = result.TotalScore / float64(result.Count)

	return result, nil
}

// EvaluateParallel runs the evaluation in parallel.
func (e *Evaluator) EvaluateParallel(ctx context.Context, module primitives.Module, dataset []*primitives.Example) (*EvaluationResult, error) {
	if len(dataset) == 0 {
		return nil, fmt.Errorf("dataset is empty")
	}

	result := &EvaluationResult{
		Scores: make([]float64, len(dataset)),
	}

	// Create worker pool
	type job struct {
		index   int
		example *primitives.Example
	}

	type jobResult struct {
		index int
		score float64
		err   error
	}

	jobs := make(chan job, len(dataset))
	results := make(chan jobResult, len(dataset))

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < e.numThreads; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				// Forward pass
				prediction, err := module.Forward(ctx, j.example.ToMap())
				if err != nil {
					results <- jobResult{index: j.index, err: err}
					continue
				}

				// Compute metric
				score := e.metric(j.example, prediction)
				results <- jobResult{index: j.index, score: score}
			}
		}()
	}

	// Send jobs
	for i, example := range dataset {
		jobs <- job{index: i, example: example}
	}
	close(jobs)

	// Wait for workers
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	for res := range results {
		if res.err != nil {
			return nil, fmt.Errorf("evaluation failed on example %d: %w", res.index, res.err)
		}
		result.Scores[res.index] = res.score
		result.TotalScore += res.score
		result.Count++
	}

	result.AverageScore = result.TotalScore / float64(result.Count)

	return result, nil
}

// WithAutoMetrics adds auto-generated metrics to the evaluator.
func (e *Evaluator) WithAutoMetrics(metrics ...*AutoMetric) *Evaluator {
	e.autoMetrics = append(e.autoMetrics, metrics...)
	return e
}

// WithLMJudge sets the LM-based judge for evaluation.
func (e *Evaluator) WithLMJudge(judge *LMJudge) *Evaluator {
	e.lmJudge = judge
	e.useLMJudge = true
	return e
}

// GenerateMetrics automatically generates evaluation metrics based on task description.
func (e *Evaluator) GenerateMetrics(ctx context.Context, lm clients.BaseLM, taskDescription string, examples []*primitives.Example) error {
	autoEvaluator := NewAutoEvaluator(lm, taskDescription)
	metrics, err := autoEvaluator.GenerateMetrics(ctx, examples)
	if err != nil {
		return fmt.Errorf("failed to generate metrics: %w", err)
	}

	e.autoMetrics = append(e.autoMetrics, metrics...)
	return nil
}
