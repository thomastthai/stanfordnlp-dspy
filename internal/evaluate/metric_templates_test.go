package evaluate

import (
	"testing"

	"github.com/stanfordnlp/dspy/internal/clients"
)

func TestMetricTemplates(t *testing.T) {
	mockLM := clients.NewMockLM("test-model")

	tests := []struct {
		name     string
		template func(clients.BaseLM) *AutoMetric
	}{
		{"Fluency", FluencyTemplate},
		{"Coherence", CoherenceTemplate},
		{"Relevance", RelevanceTemplate},
		{"Factuality", FactualityTemplate},
		{"Completeness", CompletenessTemplate},
		{"QAEvaluation", QAEvaluationTemplate},
		{"Summarization", SummarizationTemplate},
		{"Classification", ClassificationTemplate},
		{"Groundedness", GroundednessTemplate},
		{"AnswerGroundedness", AnswerGroundednessTemplate},
		{"AnswerCompleteness", AnswerCompletenessTemplate},
		{"SemanticF1", SemanticF1Template},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metric := tt.template(mockLM)

			if metric == nil {
				t.Fatal("Expected non-nil metric")
			}

			if metric.Name == "" {
				t.Error("Metric name should not be empty")
			}

			if metric.Description == "" {
				t.Error("Metric description should not be empty")
			}

			if len(metric.Aspects) == 0 {
				t.Error("Metric should have at least one aspect")
			}

			if metric.LM == nil {
				t.Error("Metric should have an LM")
			}

			if metric.Template == "" {
				t.Error("Metric template should not be empty")
			}
		})
	}
}

func TestAccuracyTemplate(t *testing.T) {
	metric := AccuracyTemplate()
	if metric != nil {
		t.Error("AccuracyTemplate should return nil (non-LM metric)")
	}
}

func TestMultiAspectTemplate(t *testing.T) {
	mockLM := clients.NewMockLM("test-model")

	aspects := []string{"fluency", "coherence", "relevance"}
	metrics := MultiAspectTemplate(mockLM, aspects)

	if len(metrics) != 3 {
		t.Errorf("Expected 3 metrics, got %d", len(metrics))
	}

	expectedNames := map[string]bool{
		"fluency":   true,
		"coherence": true,
		"relevance": true,
	}

	for _, metric := range metrics {
		if !expectedNames[metric.Name] {
			t.Errorf("Unexpected metric name: %s", metric.Name)
		}
	}
}

func TestMultiAspectTemplate_CustomAspects(t *testing.T) {
	mockLM := clients.NewMockLM("test-model")

	// Test with custom aspects that don't match predefined ones
	aspects := []string{"creativity", "originality"}
	metrics := MultiAspectTemplate(mockLM, aspects)

	if len(metrics) != 2 {
		t.Errorf("Expected 2 metrics, got %d", len(metrics))
	}

	for _, metric := range metrics {
		if metric.Name != "creativity" && metric.Name != "originality" {
			t.Errorf("Unexpected metric name: %s", metric.Name)
		}
	}
}

func TestMultiAspectTemplate_EmptyAspects(t *testing.T) {
	mockLM := clients.NewMockLM("test-model")

	aspects := []string{}
	metrics := MultiAspectTemplate(mockLM, aspects)

	if len(metrics) != 0 {
		t.Errorf("Expected 0 metrics for empty aspects, got %d", len(metrics))
	}
}
