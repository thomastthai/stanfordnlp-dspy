# Auto-Evaluation Framework Examples

This directory contains examples demonstrating the auto-evaluation and LM-as-judge capabilities in DSPy Go.

## Overview

The auto-evaluation framework enables automatic metric generation and LM-powered evaluation for complex tasks. It implements the LM-as-judge pattern, allowing language models to evaluate outputs based on various quality dimensions.

## Features

1. **LM-as-Judge**: Use language models to evaluate predictions
2. **Ensemble Judges**: Combine multiple judges for more robust evaluation
3. **Auto-Generated Metrics**: Automatically generate evaluation metrics from task descriptions
4. **Metric Templates**: Pre-built templates for common evaluation tasks
5. **Calibration Tools**: Validate metric reliability against human judgments

## Running the Examples

```bash
go run examples/auto_evaluation/main.go
```

## Example Descriptions

### Example 1: LM-as-Judge

Demonstrates basic LM-based evaluation with:
- Chain-of-thought reasoning
- Score extraction (numeric format)
- Justification generation

```go
judge := evaluate.NewLMJudge(lm, "Evaluate the quality and accuracy of the answer")
judge.WithChainOfThought(true)
judge.WithScoreFormat("numeric")

score, justification, err := judge.Judge(ctx, example, prediction)
```

### Example 2: Ensemble Judge

Shows how to combine multiple judges with different perspectives:
- Accuracy judge
- Completeness judge
- Fluency judge
- Mean aggregation

```go
ensemble := evaluate.NewEnsembleJudge(judge1, judge2, judge3)
ensemble.WithAggregation("mean")

score, justification, err := ensemble.Judge(ctx, example, prediction)
```

### Example 3: Auto-Generated Metrics

Demonstrates automatic metric generation:
- Generate metrics from task description and examples
- Evaluate using auto-generated metrics
- Multi-metric evaluation

```go
evaluator := evaluate.NewAutoEvaluator(lm, "Question answering system")
metrics, err := evaluator.GenerateMetrics(ctx, examples)
scores, err := evaluator.Evaluate(ctx, example, prediction)
```

### Example 4: Metric Templates

Shows usage of pre-built metric templates:
- Fluency template
- Coherence template
- Relevance template
- Multi-aspect evaluation

```go
fluency := evaluate.FluencyTemplate(lm)
coherence := evaluate.CoherenceTemplate(lm)
relevance := evaluate.RelevanceTemplate(lm)

metrics := evaluate.MultiAspectTemplate(lm, []string{"fluency", "coherence", "relevance"})
```

### Example 5: Calibration and Validation

Demonstrates metric reliability assessment:
- Correlation with human judgments
- Agreement rate computation
- Bias detection
- Comprehensive calibration reports

```go
calibrator := evaluate.NewMetricCalibrator()
calibrator.AddPair("ex1", humanLabel, predictedScore)

report, err := calibrator.GenerateReport()
fmt.Println(report.String())
```

## Available Score Formats

The LM-as-judge supports multiple score formats:

1. **Numeric** (default): Scores between 0 and 1
   - Example: "Score: 0.85"

2. **Letter**: Letter grades (A, B, C, D, F)
   - Example: "Grade: A"
   - A=1.0, B=0.8, C=0.6, D=0.4, F=0.0

3. **Boolean**: Yes/no judgments
   - Example: "Acceptable: yes"
   - Yes=1.0, No=0.0

## Available Metric Templates

Pre-built templates for common evaluation tasks:

### Quality Templates
- `FluencyTemplate`: Grammatical correctness and natural flow
- `CoherenceTemplate`: Logical consistency and organization
- `RelevanceTemplate`: How well output addresses the input
- `FactualityTemplate`: Factual accuracy and truthfulness
- `CompletenessTemplate`: Coverage of necessary information

### Domain-Specific Templates
- `QAEvaluationTemplate`: Question-answering quality
- `SummarizationTemplate`: Text summarization quality
- `ClassificationTemplate`: Classification quality
- `GroundednessTemplate`: Grounding in provided context

### Advanced Templates
- `AnswerGroundednessTemplate`: Answer grounding against documents
- `AnswerCompletenessTemplate`: Answer completeness against ground truth
- `SemanticF1Template`: Semantic precision and recall

## Aggregation Methods

Ensemble judges support multiple aggregation methods:

1. **Mean**: Average of all judge scores
2. **Median**: Median of all judge scores
3. **Majority**: 1.0 if majority >= 0.5, otherwise 0.0

## Calibration Metrics

The calibration framework provides:

- **Correlation**: Pearson correlation with human labels
- **Agreement**: Binary agreement rate (threshold at 0.5)
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **Bias Detection**: Over/under-estimation, variance differences

## Best Practices

1. **Use Chain-of-Thought**: Enable for more reliable evaluations
2. **Ensemble When Critical**: Use multiple judges for important decisions
3. **Calibrate Your Metrics**: Validate against human judgments
4. **Choose Appropriate Templates**: Use domain-specific templates when available
5. **Monitor Bias**: Regularly check for systematic biases in your metrics

## Integration with Existing Evaluator

The auto-evaluation framework integrates seamlessly with the existing evaluator:

```go
evaluator := evaluate.NewEvaluator(metric)

// Add LM-based judge
judge := evaluate.NewLMJudge(lm, "Evaluate quality")
evaluator.WithLMJudge(judge)

// Add auto-metrics
autoMetrics := []*evaluate.AutoMetric{...}
evaluator.WithAutoMetrics(autoMetrics...)

// Generate metrics automatically
err := evaluator.GenerateMetrics(ctx, lm, taskDescription, examples)
```

## Notes

- These examples use mock LMs for demonstration
- In production, use real LM clients (OpenAI, Anthropic, etc.)
- LM-as-judge evaluation can be expensive; use caching when appropriate
- Consider cost/quality tradeoffs when choosing models for judging
