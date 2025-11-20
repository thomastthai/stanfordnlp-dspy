package evaluate

import (
	"context"
	"testing"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/primitives"
)

func TestAutoMetric_Evaluate(t *testing.T) {
	mockLM := clients.NewMockLM("test-model")
	mockLM.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
		return &clients.Response{
			Choices: []clients.Choice{
				{
					Text: "The answer is correct and complete.\nScore: 0.95",
				},
			},
		}, nil
	}

	metric := NewAutoMetric(
		"accuracy",
		"Evaluates correctness",
		[]string{"factual accuracy", "completeness"},
		mockLM,
	)

	example := primitives.NewExample(
		map[string]interface{}{"question": "What is the capital of France?"},
		map[string]interface{}{"answer": "Paris"},
	)

	prediction := primitives.NewPrediction(
		map[string]interface{}{"answer": "Paris"},
	)

	score, explanation, err := metric.Evaluate(context.Background(), example, prediction)
	if err != nil {
		t.Fatalf("Evaluate failed: %v", err)
	}

	if score != 0.95 {
		t.Errorf("Expected score 0.95, got %f", score)
	}

	if explanation == "" {
		t.Error("Expected non-empty explanation")
	}
}

func TestAutoEvaluator_GenerateMetrics(t *testing.T) {
	mockLM := clients.NewMockLM("test-model")
	mockLM.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
		return &clients.Response{
			Choices: []clients.Choice{
				{
					Text: `METRIC: accuracy
DESCRIPTION: Measures correctness of answers
ASPECTS:
- Factual correctness
- Completeness
---
METRIC: fluency
DESCRIPTION: Evaluates language quality
ASPECTS:
- Grammar
- Natural flow`,
				},
			},
		}, nil
	}

	evaluator := NewAutoEvaluator(mockLM, "Question answering task")

	examples := []*primitives.Example{
		primitives.NewExample(
			map[string]interface{}{"question": "Test question 1"},
			map[string]interface{}{"answer": "Test answer 1"},
		),
		primitives.NewExample(
			map[string]interface{}{"question": "Test question 2"},
			map[string]interface{}{"answer": "Test answer 2"},
		),
	}

	metrics, err := evaluator.GenerateMetrics(context.Background(), examples)
	if err != nil {
		t.Fatalf("GenerateMetrics failed: %v", err)
	}

	if len(metrics) != 2 {
		t.Errorf("Expected 2 metrics, got %d", len(metrics))
	}

	if metrics[0].Name != "accuracy" {
		t.Errorf("Expected first metric name 'accuracy', got %s", metrics[0].Name)
	}

	if metrics[1].Name != "fluency" {
		t.Errorf("Expected second metric name 'fluency', got %s", metrics[1].Name)
	}
}

func TestAutoEvaluator_Evaluate(t *testing.T) {
	callCount := 0
	mockLM := clients.NewMockLM("test-model")
	mockLM.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
		callCount++
		var text string
		if callCount == 1 {
			text = "Score: 0.8"
		} else {
			text = "Score: 0.9"
		}
		return &clients.Response{
			Choices: []clients.Choice{
				{Text: text},
			},
		}, nil
	}

	evaluator := NewAutoEvaluator(mockLM, "Test task")
	
	// Add metrics manually instead of generating
	metric1 := NewAutoMetric("metric1", "First metric", []string{"aspect1"}, mockLM)
	metric2 := NewAutoMetric("metric2", "Second metric", []string{"aspect2"}, mockLM)
	
	evaluator.AddMetric(metric1)
	evaluator.AddMetric(metric2)

	example := primitives.NewExample(
		map[string]interface{}{"input": "test"},
		map[string]interface{}{"output": "expected"},
	)

	prediction := primitives.NewPrediction(
		map[string]interface{}{"output": "actual"},
	)

	scores, err := evaluator.Evaluate(context.Background(), example, prediction)
	if err != nil {
		t.Fatalf("Evaluate failed: %v", err)
	}

	if len(scores) != 2 {
		t.Errorf("Expected 2 scores, got %d", len(scores))
	}

	if scores["metric1"] != 0.8 {
		t.Errorf("Expected metric1 score 0.8, got %f", scores["metric1"])
	}

	if scores["metric2"] != 0.9 {
		t.Errorf("Expected metric2 score 0.9, got %f", scores["metric2"])
	}
}

func TestAutoEvaluator_EvaluateWithExplanations(t *testing.T) {
	mockLM := clients.NewMockLM("test-model")
	mockLM.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
		return &clients.Response{
			Choices: []clients.Choice{
				{Text: "Good quality.\nScore: 0.85"},
			},
		}, nil
	}

	evaluator := NewAutoEvaluator(mockLM, "Test task")
	metric := NewAutoMetric("quality", "Overall quality", []string{"aspect"}, mockLM)
	evaluator.AddMetric(metric)

	example := primitives.NewExample(
		map[string]interface{}{"input": "test"},
		map[string]interface{}{"output": "expected"},
	)

	prediction := primitives.NewPrediction(
		map[string]interface{}{"output": "actual"},
	)

	scores, explanations, err := evaluator.EvaluateWithExplanations(context.Background(), example, prediction)
	if err != nil {
		t.Fatalf("EvaluateWithExplanations failed: %v", err)
	}

	if len(scores) != 1 {
		t.Errorf("Expected 1 score, got %d", len(scores))
	}

	if len(explanations) != 1 {
		t.Errorf("Expected 1 explanation, got %d", len(explanations))
	}

	if explanations["quality"] == "" {
		t.Error("Expected non-empty explanation")
	}
}

func TestAutoEvaluator_AggregateScores(t *testing.T) {
	tests := []struct {
		name     string
		mode     string
		scores   map[string]float64
		expected float64
	}{
		{
			name:     "average mode",
			mode:     "average",
			scores:   map[string]float64{"m1": 0.8, "m2": 0.6, "m3": 0.9},
			expected: (0.8 + 0.6 + 0.9) / 3.0,
		},
		{
			name:     "min mode",
			mode:     "min",
			scores:   map[string]float64{"m1": 0.8, "m2": 0.6, "m3": 0.9},
			expected: 0.6,
		},
		{
			name:     "max mode",
			mode:     "max",
			scores:   map[string]float64{"m1": 0.8, "m2": 0.6, "m3": 0.9},
			expected: 0.9,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockLM := clients.NewMockLM("test-model")
			evaluator := NewAutoEvaluator(mockLM, "Test")
			evaluator.WithAggregation(tt.mode)

			result := evaluator.AggregateScores(tt.scores)
			epsilon := 0.0001
			if abs(result-tt.expected) > epsilon {
				t.Errorf("Expected %f, got %f", tt.expected, result)
			}
		})
	}
}

func TestBuildMetricTemplate(t *testing.T) {
	template := buildMetricTemplate(
		"test_metric",
		"A test metric",
		[]string{"aspect1", "aspect2"},
	)

	if template == "" {
		t.Error("Expected non-empty template")
	}

	// Check that template contains key components
	if !containsString(template, "test_metric") {
		t.Error("Template should contain metric name")
	}

	if !containsString(template, "A test metric") {
		t.Error("Template should contain description")
	}

	if !containsString(template, "aspect1") {
		t.Error("Template should contain first aspect")
	}
}

func containsString(s, substr string) bool {
	return len(s) > 0 && len(substr) > 0 && findSubstring(s, substr)
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func abs(x float64) float64 {
if x < 0 {
return -x
}
return x
}
