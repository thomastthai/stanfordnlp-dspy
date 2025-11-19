package predict

import (
	"context"
	"testing"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/primitives"
	"github.com/stanfordnlp/dspy/pkg/dspy"
)

// setupTestContext creates a test context with mock LM
func setupTestContext(t *testing.T) context.Context {
	t.Helper()

	// Create mock LM
	mockLM := clients.NewMockLM("test-model")

	// Set up response function
	mockLM.ResponseFunc = func(req *clients.Request) (*clients.Response, error) {
		// Parse the request to generate appropriate responses
		var content string
		if len(req.Messages) > 0 {
			content = generateMockResponse(req.Messages[len(req.Messages)-1].Content)
		} else {
			content = "answer: test response"
		}

		return &clients.Response{
			Choices: []clients.Choice{
				{
					Message: clients.Message{
						Role:    "assistant",
						Content: content,
					},
					Text:         content,
					Index:        0,
					FinishReason: "stop",
				},
			},
			Usage: clients.Usage{
				PromptTokens:     10,
				CompletionTokens: 20,
				TotalTokens:      30,
			},
			Model: "test-model",
			ID:    "test-id",
		}, nil
	}

	// Configure global settings with mock LM for testing
	originalSettings := dspy.GetSettings().Copy()
	t.Cleanup(func() {
		dspy.SetSettings(originalSettings)
	})

	dspy.Configure(
		dspy.WithLM(mockLM),
		dspy.WithTemperature(0.7),
		dspy.WithMaxTokens(150),
	)

	return context.Background()
}

// generateMockResponse generates a mock response based on the prompt
func generateMockResponse(prompt string) string {
	// Simple pattern matching for different module types
	if contains(prompt, "generated_code") || contains(prompt, "Python") {
		return "Reasoning: Let me write code to solve this.\nCode: def solve():\n    return 42\n\nsolve()"
	}
	if contains(prompt, "code_output") {
		return "Reasoning: Based on the code output, I can extract the answer.\nAnswer: 42"
	}
	if contains(prompt, "reasoning_attempt") {
		return "Accurate Reasoning: After comparing all attempts, the correct answer is...\nAnswer: final answer"
	}
	if contains(prompt, "reasoning") {
		return "Reasoning: Let me think step by step...\nAnswer: test answer"
	}
	return "Answer: test response"
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) &&
		(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr ||
			findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// TestProgramOfThought_New tests creation of ProgramOfThought module
func TestProgramOfThought_New(t *testing.T) {
	tests := []struct {
		name      string
		signature interface{}
		maxIters  int
		wantErr   bool
	}{
		{
			name:      "valid string signature",
			signature: "question -> answer",
			maxIters:  3,
			wantErr:   false,
		},
		{
			name:      "default max iters",
			signature: "input -> output",
			maxIters:  0,
			wantErr:   false,
		},
		{
			name:      "invalid signature type",
			signature: 123,
			maxIters:  3,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pot, err := NewProgramOfThought(tt.signature, tt.maxIters)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewProgramOfThought() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if pot == nil {
					t.Error("expected non-nil ProgramOfThought")
				}
				if pot.MaxIters <= 0 {
					t.Error("expected positive MaxIters")
				}
				if pot.CodeGenerate == nil {
					t.Error("expected non-nil CodeGenerate module")
				}
				if pot.CodeRegenerate == nil {
					t.Error("expected non-nil CodeRegenerate module")
				}
				if pot.GenerateAnswer == nil {
					t.Error("expected non-nil GenerateAnswer module")
				}
			}
		})
	}
}

// TestProgramOfThought_Forward tests the forward pass
func TestProgramOfThought_Forward(t *testing.T) {
	ctx := setupTestContext(t)

	pot, err := NewProgramOfThought("question -> answer", 3)
	if err != nil {
		t.Fatalf("failed to create ProgramOfThought: %v", err)
	}

	inputs := map[string]interface{}{
		"question": "What is 2 + 2?",
	}

	pred, err := pot.Forward(ctx, inputs)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}

	if pred == nil {
		t.Fatal("expected non-nil prediction")
	}

	// Check metadata
	if _, ok := pred.GetMetadata("generated_code"); !ok {
		t.Error("expected generated_code in metadata")
	}
	if _, ok := pred.GetMetadata("code_output"); !ok {
		t.Error("expected code_output in metadata")
	}
	if attempts, ok := pred.GetMetadata("code_attempts"); !ok || attempts.(int) < 1 {
		t.Error("expected valid code_attempts in metadata")
	}
}

// TestProgramOfThought_Copy tests module copying
func TestProgramOfThought_Copy(t *testing.T) {
	pot, err := NewProgramOfThought("input -> output", 3)
	if err != nil {
		t.Fatalf("failed to create ProgramOfThought: %v", err)
	}

	copied := pot.Copy()
	if copied == nil {
		t.Fatal("expected non-nil copy")
	}

	copiedPOT, ok := copied.(*ProgramOfThought)
	if !ok {
		t.Fatal("copy is not *ProgramOfThought")
	}

	if copiedPOT.MaxIters != pot.MaxIters {
		t.Error("MaxIters not copied correctly")
	}
}

// TestMultiChainComparison_New tests creation of MultiChainComparison module
func TestMultiChainComparison_New(t *testing.T) {
	tests := []struct {
		name      string
		signature interface{}
		m         int
		wantErr   bool
	}{
		{
			name:      "valid string signature",
			signature: "question -> answer",
			m:         3,
			wantErr:   false,
		},
		{
			name:      "default m value",
			signature: "input -> output",
			m:         0,
			wantErr:   false,
		},
		{
			name:      "invalid signature type",
			signature: []int{1, 2, 3},
			m:         3,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mcc, err := NewMultiChainComparison(tt.signature, tt.m)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewMultiChainComparison() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if mcc == nil {
					t.Error("expected non-nil MultiChainComparison")
				}
				if mcc.M <= 0 {
					t.Error("expected positive M")
				}
				// Check that reasoning attempt fields were added
				expectedFields := tt.m
				if tt.m == 0 {
					expectedFields = 3 // default
				}

				// Count reasoning_attempt fields
				count := 0
				for _, field := range mcc.Signature.InputFields {
					if len(field.Name) > 17 && field.Name[:17] == "reasoning_attempt" {
						count++
					}
				}
				if count != expectedFields {
					t.Errorf("expected %d reasoning_attempt fields, got %d", expectedFields, count)
				}
			}
		})
	}
}

// TestMultiChainComparison_Forward tests the forward pass
func TestMultiChainComparison_Forward(t *testing.T) {
	ctx := setupTestContext(t)

	mcc, err := NewMultiChainComparison("question -> answer", 2)
	if err != nil {
		t.Fatalf("failed to create MultiChainComparison: %v", err)
	}

	// Create mock completions
	completions := []*primitives.Prediction{
		primitives.NewPrediction(map[string]interface{}{
			"reasoning": "First reasoning chain",
			"answer":    "answer1",
		}),
		primitives.NewPrediction(map[string]interface{}{
			"reasoning": "Second reasoning chain",
			"answer":    "answer2",
		}),
	}

	inputs := map[string]interface{}{
		"question":    "What is the capital of France?",
		"completions": completions,
	}

	pred, err := mcc.Forward(ctx, inputs)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}

	if pred == nil {
		t.Fatal("expected non-nil prediction")
	}

	// Check metadata
	if numChains, ok := pred.GetMetadata("num_chains"); !ok || numChains.(int) != 2 {
		t.Error("expected num_chains=2 in metadata")
	}
	if method, ok := pred.GetMetadata("comparison_method"); !ok || method.(string) != "multi_chain" {
		t.Error("expected comparison_method=multi_chain in metadata")
	}
}

// TestMultiChainComparison_Forward_Errors tests error handling
func TestMultiChainComparison_Forward_Errors(t *testing.T) {
	ctx := setupTestContext(t)

	mcc, err := NewMultiChainComparison("question -> answer", 2)
	if err != nil {
		t.Fatalf("failed to create MultiChainComparison: %v", err)
	}

	tests := []struct {
		name    string
		inputs  map[string]interface{}
		wantErr bool
	}{
		{
			name:    "missing completions",
			inputs:  map[string]interface{}{"question": "test"},
			wantErr: true,
		},
		{
			name: "wrong completions type",
			inputs: map[string]interface{}{
				"question":    "test",
				"completions": "not a slice",
			},
			wantErr: true,
		},
		{
			name: "wrong number of completions",
			inputs: map[string]interface{}{
				"question": "test",
				"completions": []*primitives.Prediction{
					primitives.NewPrediction(map[string]interface{}{"answer": "a1"}),
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := mcc.Forward(ctx, tt.inputs)
			if (err != nil) != tt.wantErr {
				t.Errorf("Forward() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestMultiChainComparison_Copy tests module copying
func TestMultiChainComparison_Copy(t *testing.T) {
	mcc, err := NewMultiChainComparison("input -> output", 3)
	if err != nil {
		t.Fatalf("failed to create MultiChainComparison: %v", err)
	}

	copied := mcc.Copy()
	if copied == nil {
		t.Fatal("expected non-nil copy")
	}

	copiedMCC, ok := copied.(*MultiChainComparison)
	if !ok {
		t.Fatal("copy is not *MultiChainComparison")
	}

	if copiedMCC.M != mcc.M {
		t.Error("M not copied correctly")
	}
	if copiedMCC.Temperature != mcc.Temperature {
		t.Error("Temperature not copied correctly")
	}
}

// TestAggregation_New tests creation of Aggregation module
func TestAggregation_New(t *testing.T) {
	strategies := []string{"majority", "weighted", "consensus"}

	for _, strategy := range strategies {
		t.Run(strategy, func(t *testing.T) {
			agg := NewAggregation(strategy)
			if agg == nil {
				t.Fatal("expected non-nil Aggregation")
			}
			if agg.Strategy != strategy {
				t.Errorf("expected strategy %s, got %s", strategy, agg.Strategy)
			}
			if agg.NormalizeFunc == nil {
				t.Error("expected non-nil NormalizeFunc")
			}
		})
	}
}

// TestAggregation_MajorityVote tests majority voting
func TestAggregation_MajorityVote(t *testing.T) {
	ctx := context.Background()

	agg := NewAggregation("majority")

	predictions := []*primitives.Prediction{
		primitives.NewPrediction(map[string]interface{}{"answer": "Paris"}),
		primitives.NewPrediction(map[string]interface{}{"answer": "paris"}), // should match after normalization
		primitives.NewPrediction(map[string]interface{}{"answer": "London"}),
	}

	inputs := map[string]interface{}{
		"predictions": predictions,
		"field":       "answer",
	}

	pred, err := agg.Forward(ctx, inputs)
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}

	if pred == nil {
		t.Fatal("expected non-nil prediction")
	}

	// Check that Paris won (2 votes vs 1)
	answer, ok := pred.Get("answer")
	if !ok {
		t.Fatal("expected answer field in prediction")
	}

	answerStr, ok := answer.(string)
	if !ok {
		t.Fatal("expected answer to be string")
	}

	// Normalize and check
	normalized := agg.NormalizeFunc(answerStr)
	if normalized != "paris" {
		t.Errorf("expected 'paris', got '%s'", normalized)
	}

	// Check metadata
	if count, ok := pred.GetMetadata("majority_count"); !ok || count.(int) != 2 {
		t.Error("expected majority_count=2 in metadata")
	}
	if total, ok := pred.GetMetadata("total_predictions"); !ok || total.(int) != 3 {
		t.Error("expected total_predictions=3 in metadata")
	}
}

// TestAggregation_ConsensusVote tests consensus voting
func TestAggregation_ConsensusVote(t *testing.T) {
	ctx := context.Background()

	agg := NewAggregation("consensus")

	t.Run("with consensus", func(t *testing.T) {
		predictions := []*primitives.Prediction{
			primitives.NewPrediction(map[string]interface{}{"answer": "Paris"}),
			primitives.NewPrediction(map[string]interface{}{"answer": "paris"}),
			primitives.NewPrediction(map[string]interface{}{"answer": "PARIS"}),
		}

		inputs := map[string]interface{}{
			"predictions": predictions,
			"field":       "answer",
		}

		pred, err := agg.Forward(ctx, inputs)
		if err != nil {
			t.Fatalf("Forward() error = %v", err)
		}

		if consensus, ok := pred.GetMetadata("consensus"); !ok || consensus.(bool) != true {
			t.Error("expected consensus=true in metadata")
		}
	})

	t.Run("without consensus", func(t *testing.T) {
		predictions := []*primitives.Prediction{
			primitives.NewPrediction(map[string]interface{}{"answer": "Paris"}),
			primitives.NewPrediction(map[string]interface{}{"answer": "London"}),
		}

		inputs := map[string]interface{}{
			"predictions": predictions,
			"field":       "answer",
		}

		_, err := agg.Forward(ctx, inputs)
		if err == nil {
			t.Error("expected error for no consensus")
		}
	})
}

// TestAggregation_Forward_Errors tests error handling
func TestAggregation_Forward_Errors(t *testing.T) {
	ctx := context.Background()

	agg := NewAggregation("majority")

	tests := []struct {
		name    string
		inputs  map[string]interface{}
		wantErr bool
	}{
		{
			name:    "missing predictions",
			inputs:  map[string]interface{}{},
			wantErr: true,
		},
		{
			name: "wrong predictions type",
			inputs: map[string]interface{}{
				"predictions": "not a slice",
			},
			wantErr: true,
		},
		{
			name: "empty predictions",
			inputs: map[string]interface{}{
				"predictions": []*primitives.Prediction{},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := agg.Forward(ctx, tt.inputs)
			if (err != nil) != tt.wantErr {
				t.Errorf("Forward() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestAggregation_UnknownStrategy tests error for unknown strategy
func TestAggregation_UnknownStrategy(t *testing.T) {
	ctx := context.Background()

	agg := NewAggregation("unknown_strategy")

	predictions := []*primitives.Prediction{
		primitives.NewPrediction(map[string]interface{}{"answer": "test"}),
	}

	inputs := map[string]interface{}{
		"predictions": predictions,
	}

	_, err := agg.Forward(ctx, inputs)
	if err == nil {
		t.Error("expected error for unknown strategy")
	}
}

// TestAggregation_Copy tests module copying
func TestAggregation_Copy(t *testing.T) {
	agg := NewAggregation("majority")
	agg.Config["test"] = "value"

	copied := agg.Copy()
	if copied == nil {
		t.Fatal("expected non-nil copy")
	}

	copiedAgg, ok := copied.(*Aggregation)
	if !ok {
		t.Fatal("copy is not *Aggregation")
	}

	if copiedAgg.Strategy != agg.Strategy {
		t.Error("Strategy not copied correctly")
	}

	// Modify copy and ensure original is unchanged
	copiedAgg.Config["test"] = "modified"
	if agg.Config["test"] != "value" {
		t.Error("modifying copy affected original")
	}
}
