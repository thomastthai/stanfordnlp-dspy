package evaluate

import (
	"context"
	"testing"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/primitives"
)

func TestLMJudge_Judge(t *testing.T) {
	mockLM := clients.NewMockLM("test-model")
	mockLM.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
		return &clients.Response{
			Choices: []clients.Choice{
				{
					Text: "The output is accurate and well-structured.\nScore: 0.85",
				},
			},
		}, nil
	}

	judge := NewLMJudge(mockLM, "Evaluate the quality of the output")
	judge.WithChainOfThought(true)
	judge.WithScoreFormat("numeric")

	example := primitives.NewExample(
		map[string]interface{}{"question": "What is 2+2?"},
		map[string]interface{}{"answer": "4"},
	)

	prediction := primitives.NewPrediction(
		map[string]interface{}{"answer": "4"},
	)

	score, justification, err := judge.Judge(context.Background(), example, prediction)
	if err != nil {
		t.Fatalf("Judge failed: %v", err)
	}

	if score != 0.85 {
		t.Errorf("Expected score 0.85, got %f", score)
	}

	if justification == "" {
		t.Error("Expected non-empty justification")
	}
}

func TestLMJudge_ScoreFormats(t *testing.T) {
	tests := []struct {
		name          string
		format        string
		response      string
		expectedScore float64
	}{
		{
			name:          "numeric format",
			format:        "numeric",
			response:      "Score: 0.75",
			expectedScore: 0.75,
		},
		{
			name:          "letter format A",
			format:        "letter",
			response:      "Grade: A",
			expectedScore: 1.0,
		},
		{
			name:          "letter format B",
			format:        "letter",
			response:      "Grade: B",
			expectedScore: 0.8,
		},
		{
			name:          "boolean format yes",
			format:        "boolean",
			response:      "Acceptable: yes",
			expectedScore: 1.0,
		},
		{
			name:          "boolean format no",
			format:        "boolean",
			response:      "Acceptable: no",
			expectedScore: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockLM := clients.NewMockLM("test-model")
			response := tt.response
			mockLM.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
				return &clients.Response{
					Choices: []clients.Choice{
						{Text: response},
					},
				}, nil
			}

			judge := NewLMJudge(mockLM, "Test prompt")
			judge.WithScoreFormat(tt.format)

			example := primitives.NewExample(
				map[string]interface{}{"input": "test"},
				map[string]interface{}{"output": "expected"},
			)

			prediction := primitives.NewPrediction(
				map[string]interface{}{"output": "actual"},
			)

			score, _, err := judge.Judge(context.Background(), example, prediction)
			if err != nil {
				t.Fatalf("Judge failed: %v", err)
			}

			if score != tt.expectedScore {
				t.Errorf("Expected score %f, got %f", tt.expectedScore, score)
			}
		})
	}
}

func TestEnsembleJudge_Mean(t *testing.T) {
	// Create multiple judges with different scores
	judges := []*LMJudge{}

	scores := []float64{0.8, 0.9, 0.7}
	for _, s := range scores {
		mockLM := clients.NewMockLM("test-model")
		score := s
		mockLM.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
			return &clients.Response{
				Choices: []clients.Choice{
					{Text: "Score: " + floatToString(score)},
				},
			}, nil
		}
		judge := NewLMJudge(mockLM, "Test")
		judges = append(judges, judge)
	}

	ensemble := NewEnsembleJudge(judges...)
	ensemble.WithAggregation("mean")

	example := primitives.NewExample(
		map[string]interface{}{"input": "test"},
		map[string]interface{}{"output": "expected"},
	)

	prediction := primitives.NewPrediction(
		map[string]interface{}{"output": "actual"},
	)

	score, _, err := ensemble.Judge(context.Background(), example, prediction)
	if err != nil {
		t.Fatalf("Ensemble judge failed: %v", err)
	}

	expectedScore := (0.8 + 0.9 + 0.7) / 3.0
	epsilon := 0.0001
	if absFloat(score-expectedScore) > epsilon {
		t.Errorf("Expected mean score %f, got %f", expectedScore, score)
	}
}

func TestEnsembleJudge_Median(t *testing.T) {
	judges := []*LMJudge{}
	scores := []float64{0.6, 0.8, 0.9}

	for _, s := range scores {
		mockLM := clients.NewMockLM("test-model")
		score := s
		mockLM.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
			return &clients.Response{
				Choices: []clients.Choice{
					{Text: "Score: " + floatToString(score)},
				},
			}, nil
		}
		judge := NewLMJudge(mockLM, "Test")
		judges = append(judges, judge)
	}

	ensemble := NewEnsembleJudge(judges...)
	ensemble.WithAggregation("median")

	example := primitives.NewExample(
		map[string]interface{}{"input": "test"},
		map[string]interface{}{"output": "expected"},
	)

	prediction := primitives.NewPrediction(
		map[string]interface{}{"output": "actual"},
	)

	score, _, err := ensemble.Judge(context.Background(), example, prediction)
	if err != nil {
		t.Fatalf("Ensemble judge failed: %v", err)
	}

	expectedScore := 0.8 // median of [0.6, 0.8, 0.9]
	if score != expectedScore {
		t.Errorf("Expected median score %f, got %f", expectedScore, score)
	}
}

func TestEnsembleJudge_Majority(t *testing.T) {
	judges := []*LMJudge{}
	scores := []float64{0.6, 0.7, 0.3} // majority >= 0.5

	for _, s := range scores {
		mockLM := clients.NewMockLM("test-model")
		score := s
		mockLM.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
			return &clients.Response{
				Choices: []clients.Choice{
					{Text: "Score: " + floatToString(score)},
				},
			}, nil
		}
		judge := NewLMJudge(mockLM, "Test")
		judges = append(judges, judge)
	}

	ensemble := NewEnsembleJudge(judges...)
	ensemble.WithAggregation("majority")

	example := primitives.NewExample(
		map[string]interface{}{"input": "test"},
		map[string]interface{}{"output": "expected"},
	)

	prediction := primitives.NewPrediction(
		map[string]interface{}{"output": "actual"},
	)

	score, _, err := ensemble.Judge(context.Background(), example, prediction)
	if err != nil {
		t.Fatalf("Ensemble judge failed: %v", err)
	}

	expectedScore := 1.0 // majority pass
	if score != expectedScore {
		t.Errorf("Expected majority score %f, got %f", expectedScore, score)
	}
}

func floatToString(f float64) string {
	if f == 0.8 {
		return "0.8"
	} else if f == 0.9 {
		return "0.9"
	} else if f == 0.7 {
		return "0.7"
	} else if f == 0.6 {
		return "0.6"
	} else if f == 0.3 {
		return "0.3"
	}
	return "0.5"
}

func absFloat(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
