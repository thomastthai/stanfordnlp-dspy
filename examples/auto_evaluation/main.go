package main

import (
	"context"
	"fmt"
	"log"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/evaluate"
	"github.com/stanfordnlp/dspy/internal/primitives"
)

func main() {
	fmt.Println("=== Auto-Evaluation Framework Examples ===\n")

	// Example 1: Using LM-as-Judge for evaluation
	example1_LMJudge()

	// Example 2: Using Ensemble Judge
	example2_EnsembleJudge()

	// Example 3: Using Auto-Generated Metrics
	example3_AutoMetrics()

	// Example 4: Using Metric Templates
	example4_MetricTemplates()

	// Example 5: Calibration and Validation
	example5_Calibration()
}

func example1_LMJudge() {
	fmt.Println("Example 1: LM-as-Judge")
	fmt.Println("----------------------")

	// Create a mock LM for demonstration
	mockLM := clients.NewMockLM("gpt-4")
	mockLM.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
		return &clients.Response{
			Choices: []clients.Choice{
				{
					Text: "The answer is factually correct and well-explained.\nScore: 0.9",
				},
			},
		}, nil
	}

	// Create an LM judge
	judge := evaluate.NewLMJudge(mockLM, "Evaluate the quality and accuracy of the answer")
	judge.WithChainOfThought(true)
	judge.WithScoreFormat("numeric")

	// Create example and prediction
	example := primitives.NewExample(
		map[string]interface{}{"question": "What is the capital of France?"},
		map[string]interface{}{"answer": "Paris"},
	)

	prediction := primitives.NewPrediction(
		map[string]interface{}{"answer": "Paris is the capital of France"},
	)

	// Evaluate
	score, justification, err := judge.Judge(context.Background(), example, prediction)
	if err != nil {
		log.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Score: %.2f\n", score)
	fmt.Printf("Justification: %s\n\n", justification)
}

func example2_EnsembleJudge() {
	fmt.Println("Example 2: Ensemble Judge with Multiple Judges")
	fmt.Println("-----------------------------------------------")

	// Create multiple judges with different perspectives
	judges := []*evaluate.LMJudge{}

	// Judge 1: Accuracy focus
	mockLM1 := clients.NewMockLM("gpt-4")
	mockLM1.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
		return &clients.Response{
			Choices: []clients.Choice{{Text: "Score: 0.85"}},
		}, nil
	}
	judge1 := evaluate.NewLMJudge(mockLM1, "Evaluate factual accuracy")
	judges = append(judges, judge1)

	// Judge 2: Completeness focus
	mockLM2 := clients.NewMockLM("gpt-4")
	mockLM2.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
		return &clients.Response{
			Choices: []clients.Choice{{Text: "Score: 0.90"}},
		}, nil
	}
	judge2 := evaluate.NewLMJudge(mockLM2, "Evaluate completeness")
	judges = append(judges, judge2)

	// Judge 3: Fluency focus
	mockLM3 := clients.NewMockLM("gpt-4")
	mockLM3.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
		return &clients.Response{
			Choices: []clients.Choice{{Text: "Score: 0.80"}},
		}, nil
	}
	judge3 := evaluate.NewLMJudge(mockLM3, "Evaluate language fluency")
	judges = append(judges, judge3)

	// Create ensemble
	ensemble := evaluate.NewEnsembleJudge(judges...)
	ensemble.WithAggregation("mean")

	example := primitives.NewExample(
		map[string]interface{}{"question": "Explain photosynthesis"},
		map[string]interface{}{"answer": "Photosynthesis is the process by which plants convert light into energy"},
	)

	prediction := primitives.NewPrediction(
		map[string]interface{}{"answer": "Plants use sunlight to make food through photosynthesis"},
	)

	score, _, err := ensemble.Judge(context.Background(), example, prediction)
	if err != nil {
		log.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Ensemble Score (mean): %.2f\n", score)
	fmt.Printf("Individual judges provided their perspectives\n\n")
}

func example3_AutoMetrics() {
	fmt.Println("Example 3: Auto-Generated Metrics")
	fmt.Println("----------------------------------")

	callCount := 0
	mockLM := clients.NewMockLM("gpt-4")
	mockLM.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
		callCount++
		// First call: Generate metrics
		if callCount == 1 {
			return &clients.Response{
				Choices: []clients.Choice{
					{
						Text: `METRIC: qa_accuracy
DESCRIPTION: Evaluates the correctness of question-answering outputs
ASPECTS:
- Factual accuracy
- Relevance to question
---
METRIC: qa_completeness
DESCRIPTION: Evaluates completeness of answers
ASPECTS:
- All key points covered
- Sufficient detail`,
					},
				},
			}, nil
		}
		// Subsequent calls: Evaluation scores
		return &clients.Response{
			Choices: []clients.Choice{{Text: "The answer is good.\nScore: 0.88"}},
		}, nil
	}

	// Create auto-evaluator
	evaluator := evaluate.NewAutoEvaluator(mockLM, "Question answering system")

	// Generate metrics from examples
	examples := []*primitives.Example{
		primitives.NewExample(
			map[string]interface{}{"question": "What is AI?"},
			map[string]interface{}{"answer": "Artificial Intelligence"},
		),
		primitives.NewExample(
			map[string]interface{}{"question": "What is ML?"},
			map[string]interface{}{"answer": "Machine Learning"},
		),
	}

	metrics, err := evaluator.GenerateMetrics(context.Background(), examples)
	if err != nil {
		log.Printf("Error generating metrics: %v\n", err)
		return
	}

	fmt.Printf("Generated %d metrics:\n", len(metrics))
	for _, metric := range metrics {
		fmt.Printf("  - %s: %s\n", metric.Name, metric.Description)
	}

	// Evaluate with auto-generated metrics
	prediction := primitives.NewPrediction(
		map[string]interface{}{"answer": "AI stands for Artificial Intelligence"},
	)

	scores, err := evaluator.Evaluate(context.Background(), examples[0], prediction)
	if err != nil {
		log.Printf("Error evaluating: %v\n", err)
		return
	}

	fmt.Println("\nEvaluation scores:")
	for name, score := range scores {
		fmt.Printf("  %s: %.2f\n", name, score)
	}
	fmt.Println()
}

func example4_MetricTemplates() {
	fmt.Println("Example 4: Using Metric Templates")
	fmt.Println("----------------------------------")

	mockLM := clients.NewMockLM("gpt-4")
	mockLM.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
		return &clients.Response{
			Choices: []clients.Choice{{Text: "Score: 0.87"}},
		}, nil
	}

	// Create metrics from templates
	fluency := evaluate.FluencyTemplate(mockLM)
	coherence := evaluate.CoherenceTemplate(mockLM)
	relevance := evaluate.RelevanceTemplate(mockLM)

	fmt.Println("Available metric templates:")
	fmt.Printf("  - Fluency: %s\n", fluency.Description)
	fmt.Printf("  - Coherence: %s\n", coherence.Description)
	fmt.Printf("  - Relevance: %s\n", relevance.Description)

	// Use multi-aspect template
	aspects := []string{"fluency", "coherence", "relevance"}
	metrics := evaluate.MultiAspectTemplate(mockLM, aspects)

	fmt.Printf("\nMulti-aspect evaluation with %d metrics\n", len(metrics))

	// Evaluate using templates
	example := primitives.NewExample(
		map[string]interface{}{"input": "Write about climate change"},
		map[string]interface{}{"output": "Climate change is a global issue..."},
	)

	prediction := primitives.NewPrediction(
		map[string]interface{}{"output": "Climate change affects our planet in many ways..."},
	)

	for _, metric := range metrics {
		score, _, err := metric.Evaluate(context.Background(), example, prediction)
		if err != nil {
			log.Printf("Error: %v\n", err)
			continue
		}
		fmt.Printf("  %s: %.2f\n", metric.Name, score)
	}
	fmt.Println()
}

func example5_Calibration() {
	fmt.Println("Example 5: Metric Calibration and Validation")
	fmt.Println("---------------------------------------------")

	// Create calibrator
	calibrator := evaluate.NewMetricCalibrator()

	// Add human labels and predicted scores
	calibrator.AddPair("ex1", 0.9, 0.85)
	calibrator.AddPair("ex2", 0.7, 0.75)
	calibrator.AddPair("ex3", 0.8, 0.80)
	calibrator.AddPair("ex4", 0.6, 0.65)

	// Generate calibration report
	report, err := calibrator.GenerateReport()
	if err != nil {
		log.Printf("Error generating report: %v\n", err)
		return
	}

	fmt.Println(report.String())
	fmt.Println()
}
