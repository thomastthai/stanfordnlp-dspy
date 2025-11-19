package predict

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/primitives"
)

// Reflection performs iterative self-critique and refinement of responses.
// It generates an initial response, evaluates it, and refines it based on critique
// until convergence or max iterations is reached.
type Reflection struct {
	*primitives.BaseModule
	maxIterations        int
	critiqueAspects      []string
	convergenceThreshold float64
	reflectionHistory    []ReflectionStep
	basePredictor        primitives.Module
}

// ReflectionStep represents a single iteration in the reflection process.
type ReflectionStep struct {
	Iteration int
	Response  string
	Critique  string
	Score     float64
	Aspects   map[string]float64
	Improved  bool
}

// ReflectionOptions configures a Reflection module.
type ReflectionOptions struct {
	// MaxIterations is the maximum number of refinement iterations
	MaxIterations int

	// CritiqueAspects are the aspects to evaluate (e.g., "accuracy", "clarity")
	CritiqueAspects []string

	// ConvergenceThreshold is the minimum improvement score to continue iterating
	ConvergenceThreshold float64
}

// NewReflection creates a new Reflection module that wraps a base predictor.
// The base predictor generates initial and refined responses.
func NewReflection(basePredictor primitives.Module, opts ReflectionOptions) *Reflection {
	if opts.MaxIterations == 0 {
		opts.MaxIterations = 3
	}
	if opts.ConvergenceThreshold == 0 {
		opts.ConvergenceThreshold = 0.05
	}
	if len(opts.CritiqueAspects) == 0 {
		opts.CritiqueAspects = []string{"accuracy", "clarity", "completeness"}
	}

	return &Reflection{
		BaseModule:           primitives.NewBaseModule(),
		maxIterations:        opts.MaxIterations,
		critiqueAspects:      opts.CritiqueAspects,
		convergenceThreshold: opts.ConvergenceThreshold,
		reflectionHistory:    make([]ReflectionStep, 0),
		basePredictor:        basePredictor,
	}
}

// Forward executes the reflection loop, iteratively refining the response.
func (r *Reflection) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	// Create LM integration helper
	lmi, err := NewLMIntegration(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create LM integration: %w", err)
	}

	// Generate initial response using base predictor
	response, err := r.basePredictor.Forward(ctx, inputs)
	if err != nil {
		return nil, fmt.Errorf("initial prediction failed: %w", err)
	}

	bestResponse := response
	bestScore := 0.0
	r.reflectionHistory = []ReflectionStep{}

	// Iterative refinement loop
	for iteration := 0; iteration < r.maxIterations; iteration++ {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return bestResponse, ctx.Err()
		default:
		}

		// Generate critique using LM
		critique, aspectScores, err := r.generateCritique(ctx, lmi, inputs, response)
		if err != nil {
			// If critique fails, return best response so far
			break
		}

		// Calculate overall score
		score := calculateAverageScore(aspectScores)

		// Check if this is an improvement
		improved := score > bestScore

		// Track reflection step
		step := ReflectionStep{
			Iteration: iteration,
			Response:  extractResponse(response),
			Critique:  critique,
			Score:     score,
			Aspects:   aspectScores,
			Improved:  improved,
		}
		r.reflectionHistory = append(r.reflectionHistory, step)

		// Update best response if improved
		if improved {
			bestScore = score
			bestResponse = response
		}

		// Check convergence
		if iteration > 0 {
			improvement := score - r.reflectionHistory[iteration-1].Score
			if improvement < r.convergenceThreshold {
				break // Converged - not improving enough
			}
		}

		// Refine response based on critique
		response, err = r.refineResponse(ctx, inputs, response, critique)
		if err != nil {
			// If refinement fails, return best response so far
			break
		}
	}

	// Add reflection metadata to best response
	bestResponse.SetMetadata("reflection_history", r.reflectionHistory)
	bestResponse.SetMetadata("final_score", bestScore)
	bestResponse.SetMetadata("iterations", len(r.reflectionHistory))

	return bestResponse, nil
}

// generateCritique asks the LM to evaluate the response on specified aspects.
func (r *Reflection) generateCritique(ctx context.Context, lmi *LMIntegration,
	inputs map[string]interface{}, response *primitives.Prediction) (string, map[string]float64, error) {

	// Build critique prompt
	aspectsList := strings.Join(r.critiqueAspects, ", ")
	question := extractValue(inputs, "question", "input")
	answer := extractResponse(response)

	prompt := fmt.Sprintf(`Evaluate this response on the following aspects: %s

Question: %s
Response: %s

Provide:
1. A score (0.0-1.0) for each aspect
2. Constructive feedback for improvement

Format your response as:
accuracy: [score between 0.0 and 1.0]
clarity: [score between 0.0 and 1.0]
completeness: [score between 0.0 and 1.0]
feedback: [your detailed critique]`, aspectsList, question, answer)

	request := &clients.Request{
		Messages: []clients.Message{
			{Role: "system", Content: "You are a helpful critic evaluating responses. Be constructive and specific."},
			{Role: "user", Content: prompt},
		},
		Temperature: 0.3,
		MaxTokens:   500,
	}

	resp, err := lmi.Call(ctx, request)
	if err != nil {
		return "", nil, fmt.Errorf("critique generation failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", nil, fmt.Errorf("no critique generated")
	}

	content := resp.Choices[0].Message.Content
	aspectScores := parseAspectScores(content, r.critiqueAspects)

	return content, aspectScores, nil
}

// refineResponse generates an improved response based on critique.
func (r *Reflection) refineResponse(ctx context.Context,
	inputs map[string]interface{}, previousResponse *primitives.Prediction,
	critique string) (*primitives.Prediction, error) {

	question := extractValue(inputs, "question", "input")

	// Create new inputs with refinement context
	refinementInputs := make(map[string]interface{})
	for k, v := range inputs {
		refinementInputs[k] = v
	}

	// Add refinement context to inputs
	if question != "" {
		refinementInputs["question"] = fmt.Sprintf("%s\n\nRefinement guidance:\n%s", question, critique)
	}

	// Use base predictor to generate refined response
	return r.basePredictor.Forward(ctx, refinementInputs)
}

// parseAspectScores extracts scores for each aspect from the critique content.
func parseAspectScores(content string, aspects []string) map[string]float64 {
	scores := make(map[string]float64)

	// Look for "aspect: score" pattern
	for _, aspect := range aspects {
		// Try multiple patterns
		patterns := []string{
			aspect + `:\s*([0-9]*\.?[0-9]+)`,
			aspect + `\s*:\s*([0-9]*\.?[0-9]+)`,
			aspect + `\s*-\s*([0-9]*\.?[0-9]+)`,
		}

		found := false
		for _, pattern := range patterns {
			re := regexp.MustCompile(`(?i)` + pattern)
			matches := re.FindStringSubmatch(content)
			if len(matches) > 1 {
				if score, err := strconv.ParseFloat(matches[1], 64); err == nil {
					// Clamp score to [0, 1]
					if score < 0 {
						score = 0
					} else if score > 1 {
						score = 1
					}
					scores[aspect] = score
					found = true
					break
				}
			}
		}

		// Default to middle score if not found
		if !found {
			scores[aspect] = 0.5
		}
	}

	return scores
}

// calculateAverageScore computes the average of aspect scores.
func calculateAverageScore(scores map[string]float64) float64 {
	if len(scores) == 0 {
		return 0.0
	}

	total := 0.0
	for _, score := range scores {
		total += score
	}
	return total / float64(len(scores))
}

// extractResponse gets the main response text from a prediction.
func extractResponse(pred *primitives.Prediction) string {
	// Try common output field names
	fields := []string{"answer", "response", "output", "result"}

	for _, field := range fields {
		if val, ok := pred.Get(field); ok {
			return fmt.Sprintf("%v", val)
		}
	}

	// Fallback: return first field
	for _, val := range pred.Fields() {
		return fmt.Sprintf("%v", val)
	}

	return ""
}

// extractValue gets a value from inputs, trying multiple field names.
func extractValue(inputs map[string]interface{}, fields ...string) string {
	for _, field := range fields {
		if val, ok := inputs[field]; ok {
			return fmt.Sprintf("%v", val)
		}
	}
	return ""
}

// Copy creates a deep copy of the Reflection module.
func (r *Reflection) Copy() primitives.Module {
	return &Reflection{
		BaseModule:           primitives.NewBaseModule(),
		maxIterations:        r.maxIterations,
		critiqueAspects:      append([]string{}, r.critiqueAspects...),
		convergenceThreshold: r.convergenceThreshold,
		reflectionHistory:    make([]ReflectionStep, 0),
		basePredictor:        r.basePredictor.Copy(),
	}
}

// NamedParameters returns all parameters from the base predictor.
func (r *Reflection) NamedParameters() []primitives.NamedParameter {
	return r.basePredictor.NamedParameters()
}
