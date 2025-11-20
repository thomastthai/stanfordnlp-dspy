package evaluate

import (
	"context"
	"fmt"
	"strings"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/primitives"
)

// AutoMetric represents an automatically generated evaluation metric using LM-as-judge.
type AutoMetric struct {
	Name               string
	Description        string
	Aspects            []string
	LM                 clients.BaseLM
	Template           string
	ScoreRange         [2]float64 // [min, max]
	RequireExplanation bool
	judge              *LMJudge
}

// NewAutoMetric creates a new auto-generated metric.
func NewAutoMetric(name, description string, aspects []string, lm clients.BaseLM) *AutoMetric {
	if aspects == nil {
		aspects = []string{}
	}

	// Build evaluation prompt from aspects
	template := buildMetricTemplate(name, description, aspects)

	judge := NewLMJudge(lm, template)
	judge.WithChainOfThought(true)
	judge.WithJustification(true)
	judge.WithScoreFormat("numeric")

	return &AutoMetric{
		Name:               name,
		Description:        description,
		Aspects:            aspects,
		LM:                 lm,
		Template:           template,
		ScoreRange:         [2]float64{0.0, 1.0},
		RequireExplanation: true,
		judge:              judge,
	}
}

// Evaluate evaluates a prediction against an example and returns a score and explanation.
func (a *AutoMetric) Evaluate(ctx context.Context, example *primitives.Example, prediction *primitives.Prediction) (float64, string, error) {
	return a.judge.Judge(ctx, example, prediction)
}

// buildMetricTemplate constructs an evaluation prompt from metric information.
func buildMetricTemplate(name, description string, aspects []string) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("Evaluation Metric: %s\n", name))
	sb.WriteString(fmt.Sprintf("Description: %s\n\n", description))

	if len(aspects) > 0 {
		sb.WriteString("Evaluate the following aspects:\n")
		for i, aspect := range aspects {
			sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, aspect))
		}
		sb.WriteString("\n")
	}

	sb.WriteString("Rate the quality of the system output considering all aspects listed above.\n")

	return sb.String()
}

// AutoEvaluator generates and uses automatic metrics for evaluation.
type AutoEvaluator struct {
	LM              clients.BaseLM
	TaskDescription string
	Metrics         []*AutoMetric
	UseExplanations bool
	AggregationMode string // "average", "weighted", "min", "max"
}

// NewAutoEvaluator creates a new auto-evaluator for a given task.
func NewAutoEvaluator(lm clients.BaseLM, taskDescription string) *AutoEvaluator {
	return &AutoEvaluator{
		LM:              lm,
		TaskDescription: taskDescription,
		Metrics:         []*AutoMetric{},
		UseExplanations: true,
		AggregationMode: "average",
	}
}

// WithAggregation sets the aggregation mode for combining multiple metric scores.
func (a *AutoEvaluator) WithAggregation(mode string) *AutoEvaluator {
	a.AggregationMode = mode
	return a
}

// WithExplanations enables or disables explanation generation.
func (a *AutoEvaluator) WithExplanations(use bool) *AutoEvaluator {
	a.UseExplanations = use
	return a
}

// AddMetric adds a metric to the evaluator.
func (a *AutoEvaluator) AddMetric(metric *AutoMetric) *AutoEvaluator {
	a.Metrics = append(a.Metrics, metric)
	return a
}

// GenerateMetrics automatically generates evaluation metrics based on task description and examples.
func (a *AutoEvaluator) GenerateMetrics(ctx context.Context, examples []*primitives.Example) ([]*AutoMetric, error) {
	if len(examples) == 0 {
		return nil, fmt.Errorf("no examples provided for metric generation")
	}

	// Build prompt for metric generation
	prompt := a.buildMetricGenerationPrompt(examples)

	// Call LM to generate metrics
	request := clients.NewRequest().
		WithPrompt(prompt).
		WithTemperature(0.3).
		WithMaxTokens(1000)

	response, err := a.LM.Call(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("failed to generate metrics: %w", err)
	}

	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("no response from LM")
	}

	text := response.Choices[0].Text
	if text == "" && response.Choices[0].Message.Content != "" {
		text = response.Choices[0].Message.Content
	}

	// Parse the response to extract metrics
	metrics, err := a.parseGeneratedMetrics(text)
	if err != nil {
		return nil, fmt.Errorf("failed to parse generated metrics: %w", err)
	}

	// Store generated metrics
	a.Metrics = append(a.Metrics, metrics...)

	return metrics, nil
}

// buildMetricGenerationPrompt constructs a prompt for generating evaluation metrics.
func (a *AutoEvaluator) buildMetricGenerationPrompt(examples []*primitives.Example) string {
	var sb strings.Builder

	sb.WriteString("You are an expert in designing evaluation metrics for AI systems.\n\n")
	sb.WriteString(fmt.Sprintf("Task Description: %s\n\n", a.TaskDescription))

	// Add sample examples
	sb.WriteString("Sample Examples:\n")
	numSamples := 3
	if len(examples) < numSamples {
		numSamples = len(examples)
	}

	for i := 0; i < numSamples; i++ {
		example := examples[i]
		sb.WriteString(fmt.Sprintf("\nExample %d:\n", i+1))
		sb.WriteString("Inputs:\n")
		for key, value := range example.Inputs() {
			sb.WriteString(fmt.Sprintf("  %s: %v\n", key, value))
		}
		if len(example.Outputs()) > 0 {
			sb.WriteString("Expected Outputs:\n")
			for key, value := range example.Outputs() {
				sb.WriteString(fmt.Sprintf("  %s: %v\n", key, value))
			}
		}
	}

	sb.WriteString("\nBased on the task description and examples, suggest 2-3 evaluation metrics.\n")
	sb.WriteString("For each metric, provide:\n")
	sb.WriteString("1. Metric Name (concise, lowercase with underscores)\n")
	sb.WriteString("2. Description (one sentence)\n")
	sb.WriteString("3. Aspects to evaluate (2-4 specific criteria)\n\n")
	sb.WriteString("Format your response as:\n")
	sb.WriteString("METRIC: <name>\n")
	sb.WriteString("DESCRIPTION: <description>\n")
	sb.WriteString("ASPECTS:\n")
	sb.WriteString("- <aspect 1>\n")
	sb.WriteString("- <aspect 2>\n")
	sb.WriteString("---\n")

	return sb.String()
}

// parseGeneratedMetrics parses the LM response to extract metric definitions.
func (a *AutoEvaluator) parseGeneratedMetrics(text string) ([]*AutoMetric, error) {
	metrics := []*AutoMetric{}

	// Split by metric separator
	metricBlocks := strings.Split(text, "---")

	for _, block := range metricBlocks {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}

		var name, description string
		aspects := []string{}

		lines := strings.Split(block, "\n")
		inAspects := false

		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}

			if strings.HasPrefix(strings.ToUpper(line), "METRIC:") {
				name = strings.TrimSpace(strings.TrimPrefix(strings.ToUpper(line), "METRIC:"))
				name = strings.TrimSpace(strings.TrimPrefix(name, strings.ToUpper("METRIC:")))
				name = strings.TrimSpace(line[7:])
			} else if strings.HasPrefix(strings.ToUpper(line), "DESCRIPTION:") {
				description = strings.TrimSpace(line[12:])
			} else if strings.HasPrefix(strings.ToUpper(line), "ASPECTS:") {
				inAspects = true
			} else if inAspects && strings.HasPrefix(line, "-") {
				aspect := strings.TrimSpace(strings.TrimPrefix(line, "-"))
				if aspect != "" {
					aspects = append(aspects, aspect)
				}
			}
		}

		if name != "" && description != "" {
			metric := NewAutoMetric(name, description, aspects, a.LM)
			metrics = append(metrics, metric)
		}
	}

	if len(metrics) == 0 {
		return nil, fmt.Errorf("no metrics could be parsed from response")
	}

	return metrics, nil
}

// Evaluate evaluates a prediction using all configured metrics.
// Returns a map of metric names to scores.
func (a *AutoEvaluator) Evaluate(ctx context.Context, example *primitives.Example, prediction *primitives.Prediction) (map[string]float64, error) {
	if len(a.Metrics) == 0 {
		return nil, fmt.Errorf("no metrics configured")
	}

	scores := make(map[string]float64)

	for _, metric := range a.Metrics {
		score, _, err := metric.Evaluate(ctx, example, prediction)
		if err != nil {
			return nil, fmt.Errorf("metric %s failed: %w", metric.Name, err)
		}
		scores[metric.Name] = score
	}

	return scores, nil
}

// EvaluateWithExplanations evaluates and returns both scores and explanations.
func (a *AutoEvaluator) EvaluateWithExplanations(ctx context.Context, example *primitives.Example, prediction *primitives.Prediction) (map[string]float64, map[string]string, error) {
	if len(a.Metrics) == 0 {
		return nil, nil, fmt.Errorf("no metrics configured")
	}

	scores := make(map[string]float64)
	explanations := make(map[string]string)

	for _, metric := range a.Metrics {
		score, explanation, err := metric.Evaluate(ctx, example, prediction)
		if err != nil {
			return nil, nil, fmt.Errorf("metric %s failed: %w", metric.Name, err)
		}
		scores[metric.Name] = score
		explanations[metric.Name] = explanation
	}

	return scores, explanations, nil
}

// AggregateScores combines multiple metric scores into a single score.
func (a *AutoEvaluator) AggregateScores(scores map[string]float64) float64 {
	if len(scores) == 0 {
		return 0.0
	}

	values := make([]float64, 0, len(scores))
	for _, score := range scores {
		values = append(values, score)
	}

	switch a.AggregationMode {
	case "average":
		return mean(values)
	case "min":
		return minSlice(values)
	case "max":
		return maxSlice(values)
	case "weighted":
		// For weighted, we'd need weights per metric
		// Fall back to average for now
		return mean(values)
	default:
		return mean(values)
	}
}

// minSlice returns the minimum value in a slice.
func minSlice(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}
	minVal := values[0]
	for _, v := range values[1:] {
		if v < minVal {
			minVal = v
		}
	}
	return minVal
}

// maxSlice returns the maximum value in a slice.
func maxSlice(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}
	maxVal := values[0]
	for _, v := range values[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}
