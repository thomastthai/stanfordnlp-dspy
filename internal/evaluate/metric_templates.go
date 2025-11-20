package evaluate

import (
	"github.com/stanfordnlp/dspy/internal/clients"
)

// AccuracyTemplate creates a basic accuracy metric that doesn't require an LM.
// This is kept for compatibility but returns nil since it's a non-LM metric.
func AccuracyTemplate() *AutoMetric {
	// Accuracy is typically computed without LM, so we return nil
	// Users should use the existing ExactMatch or F1Score metrics instead
	return nil
}

// FluencyTemplate creates an LM-based fluency evaluation metric.
func FluencyTemplate(lm clients.BaseLM) *AutoMetric {
	aspects := []string{
		"Grammar and syntax correctness",
		"Natural language flow",
		"Vocabulary appropriateness",
		"Absence of awkward phrasing",
	}

	return NewAutoMetric(
		"fluency",
		"Evaluates the grammatical correctness and natural flow of the generated text",
		aspects,
		lm,
	)
}

// CoherenceTemplate creates an LM-based coherence evaluation metric.
func CoherenceTemplate(lm clients.BaseLM) *AutoMetric {
	aspects := []string{
		"Logical consistency of ideas",
		"Smooth transitions between concepts",
		"Overall structural organization",
		"Internal consistency of information",
	}

	return NewAutoMetric(
		"coherence",
		"Evaluates the logical consistency and organization of the output",
		aspects,
		lm,
	)
}

// RelevanceTemplate creates an LM-based relevance evaluation metric.
func RelevanceTemplate(lm clients.BaseLM) *AutoMetric {
	aspects := []string{
		"Addresses the input query directly",
		"Stays on-topic throughout",
		"Includes pertinent information",
		"Avoids irrelevant content",
	}

	return NewAutoMetric(
		"relevance",
		"Evaluates how well the output addresses the input query",
		aspects,
		lm,
	)
}

// FactualityTemplate creates an LM-based factuality evaluation metric.
func FactualityTemplate(lm clients.BaseLM) *AutoMetric {
	aspects := []string{
		"Factual correctness of statements",
		"Absence of fabricated information",
		"Consistency with ground truth",
		"Verifiable claims",
	}

	return NewAutoMetric(
		"factuality",
		"Evaluates the factual accuracy and truthfulness of the output",
		aspects,
		lm,
	)
}

// CompletenessTemplate creates an LM-based completeness evaluation metric.
func CompletenessTemplate(lm clients.BaseLM) *AutoMetric {
	aspects := []string{
		"Coverage of all key points",
		"No critical information missing",
		"Fully addresses the question",
		"Provides sufficient detail",
	}

	return NewAutoMetric(
		"completeness",
		"Evaluates whether the output covers all necessary information",
		aspects,
		lm,
	)
}

// QAEvaluationTemplate creates a comprehensive Q&A evaluation metric.
func QAEvaluationTemplate(lm clients.BaseLM) *AutoMetric {
	aspects := []string{
		"Directly answers the question asked",
		"Provides accurate information",
		"Is complete without being verbose",
		"Uses clear and understandable language",
	}

	return NewAutoMetric(
		"qa_quality",
		"Evaluates the quality of question-answering outputs",
		aspects,
		lm,
	)
}

// SummarizationTemplate creates a summarization evaluation metric.
func SummarizationTemplate(lm clients.BaseLM) *AutoMetric {
	aspects := []string{
		"Captures main ideas from source",
		"Is concise and focused",
		"Maintains factual accuracy",
		"Avoids unnecessary details",
		"Is coherent and well-structured",
	}

	return NewAutoMetric(
		"summarization_quality",
		"Evaluates the quality of text summarization",
		aspects,
		lm,
	)
}

// ClassificationTemplate creates a classification evaluation metric.
func ClassificationTemplate(lm clients.BaseLM) *AutoMetric {
	aspects := []string{
		"Correct category assignment",
		"Appropriate confidence level",
		"Clear reasoning for classification",
		"Consistent with provided examples",
	}

	return NewAutoMetric(
		"classification_quality",
		"Evaluates the quality of classification outputs",
		aspects,
		lm,
	)
}

// MultiAspectTemplate creates multiple metrics for comprehensive evaluation.
func MultiAspectTemplate(lm clients.BaseLM, aspectNames []string) []*AutoMetric {
	metrics := []*AutoMetric{}

	// Map aspect names to predefined templates
	aspectMap := map[string]func(clients.BaseLM) *AutoMetric{
		"fluency":      FluencyTemplate,
		"coherence":    CoherenceTemplate,
		"relevance":    RelevanceTemplate,
		"factuality":   FactualityTemplate,
		"completeness": CompletenessTemplate,
	}

	for _, aspectName := range aspectNames {
		if templateFunc, ok := aspectMap[aspectName]; ok {
			if metric := templateFunc(lm); metric != nil {
				metrics = append(metrics, metric)
			}
		}
	}

	// If no aspects matched, create custom metrics
	if len(metrics) == 0 && len(aspectNames) > 0 {
		for _, aspectName := range aspectNames {
			metric := NewAutoMetric(
				aspectName,
				"Evaluates the "+aspectName+" of the output",
				[]string{"Quality of " + aspectName},
				lm,
			)
			metrics = append(metrics, metric)
		}
	}

	return metrics
}

// GroundednessTemplate creates a groundedness evaluation metric for RAG systems.
func GroundednessTemplate(lm clients.BaseLM) *AutoMetric {
	aspects := []string{
		"All claims are supported by provided context",
		"No hallucinated or fabricated information",
		"Citations or references are accurate",
		"Avoids speculation beyond provided facts",
	}

	return NewAutoMetric(
		"groundedness",
		"Evaluates whether the output is grounded in provided context/documents",
		aspects,
		lm,
	)
}

// AnswerGroundednessTemplate creates a metric for evaluating answer groundedness against context.
func AnswerGroundednessTemplate(lm clients.BaseLM) *AutoMetric {
	aspects := []string{
		"System response claims are enumerated",
		"Claims are checked against retrieved context",
		"Fraction of claims supported by context",
		"Basic commonsense reasoning is applied",
	}

	return NewAutoMetric(
		"answer_groundedness",
		"Estimates the groundedness of system responses against retrieved documents",
		aspects,
		lm,
	)
}

// AnswerCompletenessTemplate creates a metric for evaluating answer completeness.
func AnswerCompletenessTemplate(lm clients.BaseLM) *AutoMetric {
	aspects := []string{
		"Key ideas in ground truth are identified",
		"Key ideas in system response are identified",
		"Overlap between ground truth and response is discussed",
		"Fraction of ground truth covered by response",
	}

	return NewAutoMetric(
		"answer_completeness",
		"Estimates the completeness of system responses against ground truth",
		aspects,
		lm,
	)
}

// SemanticF1Template creates a metric for semantic F1 evaluation.
func SemanticF1Template(lm clients.BaseLM) *AutoMetric {
	aspects := []string{
		"Semantic precision: fraction of system response covered by ground truth",
		"Semantic recall: fraction of ground truth covered by system response",
		"F1 score combining precision and recall",
		"Key ideas enumeration and matching",
	}

	return NewAutoMetric(
		"semantic_f1",
		"Evaluates semantic precision and recall between prediction and ground truth",
		aspects,
		lm,
	)
}
