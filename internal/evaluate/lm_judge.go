package evaluate

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/primitives"
)

// LMJudge uses a language model to evaluate predictions.
type LMJudge struct {
	LM                   clients.BaseLM
	EvaluationPrompt     string
	UseChainOfThought    bool
	ScoreFormat          string // "numeric", "letter", "boolean"
	RequireJustification bool
}

// NewLMJudge creates a new LM-based judge with the given evaluation prompt.
func NewLMJudge(lm clients.BaseLM, evaluationPrompt string) *LMJudge {
	return &LMJudge{
		LM:                   lm,
		EvaluationPrompt:     evaluationPrompt,
		UseChainOfThought:    true,
		ScoreFormat:          "numeric",
		RequireJustification: true,
	}
}

// WithChainOfThought enables or disables chain-of-thought reasoning.
func (l *LMJudge) WithChainOfThought(use bool) *LMJudge {
	l.UseChainOfThought = use
	return l
}

// WithScoreFormat sets the expected score format.
func (l *LMJudge) WithScoreFormat(format string) *LMJudge {
	l.ScoreFormat = format
	return l
}

// WithJustification enables or disables justification requirement.
func (l *LMJudge) WithJustification(require bool) *LMJudge {
	l.RequireJustification = require
	return l
}

// Judge evaluates a prediction against an example using the language model.
// Returns a normalized score between 0 and 1, a justification string, and any error.
func (l *LMJudge) Judge(ctx context.Context, example *primitives.Example, prediction *primitives.Prediction) (float64, string, error) {
	// Build the evaluation prompt
	prompt := l.buildPrompt(example, prediction)

	// Call the language model
	request := clients.NewRequest().
		WithPrompt(prompt).
		WithTemperature(0.0).
		WithMaxTokens(500)

	response, err := l.LM.Call(ctx, request)
	if err != nil {
		return 0.0, "", fmt.Errorf("LM call failed: %w", err)
	}

	if len(response.Choices) == 0 {
		return 0.0, "", fmt.Errorf("no choices in LM response")
	}

	// Parse the response
	text := response.Choices[0].Text
	if text == "" && response.Choices[0].Message.Content != "" {
		text = response.Choices[0].Message.Content
	}

	score, justification, err := l.parseResponse(text)
	if err != nil {
		return 0.0, "", fmt.Errorf("failed to parse LM response: %w", err)
	}

	return score, justification, nil
}

// buildPrompt constructs the evaluation prompt from the example and prediction.
func (l *LMJudge) buildPrompt(example *primitives.Example, prediction *primitives.Prediction) string {
	var sb strings.Builder

	// Add system instruction
	sb.WriteString("You are an expert evaluator assessing the quality of system outputs.\n\n")

	// Add evaluation criteria
	sb.WriteString(l.EvaluationPrompt)
	sb.WriteString("\n\n")

	// Add example context
	sb.WriteString("Input:\n")
	for key, value := range example.Inputs() {
		sb.WriteString(fmt.Sprintf("%s: %v\n", key, value))
	}
	sb.WriteString("\n")

	// Add expected output if available
	if len(example.Outputs()) > 0 {
		sb.WriteString("Expected Output:\n")
		for key, value := range example.Outputs() {
			sb.WriteString(fmt.Sprintf("%s: %v\n", key, value))
		}
		sb.WriteString("\n")
	}

	// Add predicted output
	sb.WriteString("System Output:\n")
	for key, value := range prediction.Fields() {
		sb.WriteString(fmt.Sprintf("%s: %v\n", key, value))
	}
	sb.WriteString("\n")

	// Add instructions based on configuration
	if l.UseChainOfThought {
		sb.WriteString("Please think step by step and provide your reasoning.\n")
	}

	if l.RequireJustification {
		sb.WriteString("Provide a justification for your evaluation.\n")
	}

	// Add scoring instruction based on format
	switch l.ScoreFormat {
	case "numeric":
		sb.WriteString("Provide a score between 0 and 1, where 1 is perfect.\n")
		sb.WriteString("Format your response as: Score: <number>\n")
	case "letter":
		sb.WriteString("Provide a letter grade (A, B, C, D, or F).\n")
		sb.WriteString("Format your response as: Grade: <letter>\n")
	case "boolean":
		sb.WriteString("Evaluate if the output is acceptable (yes/no).\n")
		sb.WriteString("Format your response as: Acceptable: <yes/no>\n")
	default:
		sb.WriteString("Provide a score between 0 and 1, where 1 is perfect.\n")
		sb.WriteString("Format your response as: Score: <number>\n")
	}

	return sb.String()
}

// parseResponse extracts the score and justification from the LM response.
func (l *LMJudge) parseResponse(text string) (float64, string, error) {
	// Split into lines for easier parsing
	lines := strings.Split(text, "\n")

	var score float64
	var justification strings.Builder
	scoreFound := false

	for _, line := range lines {
		line = strings.TrimSpace(line)

		// Try to extract score based on format
		switch l.ScoreFormat {
		case "numeric":
			if match := regexp.MustCompile(`(?i)score:\s*([0-9]*\.?[0-9]+)`).FindStringSubmatch(line); len(match) > 1 {
				parsedScore, err := strconv.ParseFloat(match[1], 64)
				if err == nil {
					score = parsedScore
					scoreFound = true
					continue
				}
			}
		case "letter":
			if match := regexp.MustCompile(`(?i)grade:\s*([A-F])`).FindStringSubmatch(line); len(match) > 1 {
				score = letterToScore(match[1])
				scoreFound = true
				continue
			}
		case "boolean":
			if match := regexp.MustCompile(`(?i)acceptable:\s*(yes|no)`).FindStringSubmatch(line); len(match) > 1 {
				if strings.ToLower(match[1]) == "yes" {
					score = 1.0
				} else {
					score = 0.0
				}
				scoreFound = true
				continue
			}
		}

		// Collect justification
		if line != "" && !strings.Contains(strings.ToLower(line), "score:") &&
			!strings.Contains(strings.ToLower(line), "grade:") &&
			!strings.Contains(strings.ToLower(line), "acceptable:") {
			if justification.Len() > 0 {
				justification.WriteString(" ")
			}
			justification.WriteString(line)
		}
	}

	if !scoreFound {
		return 0.0, "", fmt.Errorf("could not find score in response")
	}

	// Normalize score to [0, 1]
	if score < 0 {
		score = 0
	} else if score > 1 {
		score = 1
	}

	return score, justification.String(), nil
}

// letterToScore converts a letter grade to a numeric score.
func letterToScore(letter string) float64 {
	switch strings.ToUpper(letter) {
	case "A":
		return 1.0
	case "B":
		return 0.8
	case "C":
		return 0.6
	case "D":
		return 0.4
	case "F":
		return 0.0
	default:
		return 0.5
	}
}

// EnsembleJudge combines multiple judges to produce a final score.
type EnsembleJudge struct {
	Judges      []*LMJudge
	Aggregation string // "mean", "median", "majority"
}

// NewEnsembleJudge creates a new ensemble judge with the given judges.
func NewEnsembleJudge(judges ...*LMJudge) *EnsembleJudge {
	return &EnsembleJudge{
		Judges:      judges,
		Aggregation: "mean",
	}
}

// WithAggregation sets the aggregation method for combining judge scores.
func (e *EnsembleJudge) WithAggregation(method string) *EnsembleJudge {
	e.Aggregation = method
	return e
}

// Judge evaluates using all judges and aggregates their scores.
func (e *EnsembleJudge) Judge(ctx context.Context, example *primitives.Example, prediction *primitives.Prediction) (float64, string, error) {
	if len(e.Judges) == 0 {
		return 0.0, "", fmt.Errorf("no judges configured")
	}

	scores := make([]float64, 0, len(e.Judges))
	justifications := make([]string, 0, len(e.Judges))

	// Collect scores from all judges
	for i, judge := range e.Judges {
		score, justification, err := judge.Judge(ctx, example, prediction)
		if err != nil {
			return 0.0, "", fmt.Errorf("judge %d failed: %w", i, err)
		}
		scores = append(scores, score)
		justifications = append(justifications, justification)
	}

	// Aggregate scores
	var finalScore float64
	switch e.Aggregation {
	case "mean":
		finalScore = mean(scores)
	case "median":
		finalScore = median(scores)
	case "majority":
		finalScore = majority(scores)
	default:
		finalScore = mean(scores)
	}

	// Combine justifications
	combinedJustification := fmt.Sprintf("Ensemble of %d judges (aggregation: %s):\n", len(e.Judges), e.Aggregation)
	for i, just := range justifications {
		combinedJustification += fmt.Sprintf("\nJudge %d (score: %.2f): %s", i+1, scores[i], just)
	}

	return finalScore, combinedJustification, nil
}

// mean calculates the arithmetic mean of a slice of scores.
func mean(scores []float64) float64 {
	if len(scores) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, score := range scores {
		sum += score
	}
	return sum / float64(len(scores))
}

// median calculates the median of a slice of scores.
func median(scores []float64) float64 {
	if len(scores) == 0 {
		return 0.0
	}

	// Create a copy and sort
	sorted := make([]float64, len(scores))
	copy(sorted, scores)

	// Simple bubble sort for small arrays
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) / 2.0
	}
	return sorted[mid]
}

// majority returns 1.0 if majority of scores are >= 0.5, otherwise 0.0.
func majority(scores []float64) float64 {
	if len(scores) == 0 {
		return 0.0
	}

	passing := 0
	for _, score := range scores {
		if score >= 0.5 {
			passing++
		}
	}

	if float64(passing) >= float64(len(scores))/2.0 {
		return 1.0
	}
	return 0.0
}
