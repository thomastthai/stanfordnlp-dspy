package teleprompt

import (
	"context"
	"testing"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

func TestLabeledFewShot_New(t *testing.T) {
	optimizer := NewLabeledFewShot(5)
	if optimizer == nil {
		t.Fatal("NewLabeledFewShot returned nil")
	}

	if optimizer.K != 5 {
		t.Errorf("expected K=5, got %d", optimizer.K)
	}

	if !optimizer.Sample {
		t.Error("expected Sample=true by default")
	}

	if optimizer.Name() != "LabeledFewShot" {
		t.Errorf("expected name 'LabeledFewShot', got '%s'", optimizer.Name())
	}
}

func TestLabeledFewShot_WithOptions(t *testing.T) {
	optimizer := NewLabeledFewShot(5).
		WithSample(false).
		WithSeed(42)

	if optimizer.Sample {
		t.Error("expected Sample=false")
	}

	if optimizer.Seed != 42 {
		t.Errorf("expected Seed=42, got %d", optimizer.Seed)
	}
}

// mockModule is a test implementation of primitives.Module
type mockModule struct {
	*primitives.BaseModule
}

func (m *mockModule) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	return primitives.NewPrediction(map[string]interface{}{"output": "test"}), nil
}

func (m *mockModule) Copy() primitives.Module {
	return &mockModule{
		BaseModule: primitives.NewBaseModule(),
	}
}

func (m *mockModule) Save() ([]byte, error) {
	return m.BaseModule.Save()
}

func (m *mockModule) Load(data []byte) error {
	return m.BaseModule.Load(data)
}

func TestLabeledFewShot_CompileEmptyTrainset(t *testing.T) {
	ctx := context.Background()
	optimizer := NewLabeledFewShot(5)

	module := &mockModule{BaseModule: primitives.NewBaseModule()}
	trainset := []*primitives.Example{}

	compiled, err := optimizer.Compile(ctx, module, trainset, nil)
	if err != nil {
		t.Fatalf("Compile failed with empty trainset: %v", err)
	}

	if compiled == nil {
		t.Fatal("Compile returned nil module")
	}
}

func TestLabeledFewShot_CompileWithExamples(t *testing.T) {
	ctx := context.Background()
	optimizer := NewLabeledFewShot(3)

	// Create a test module
	module := &mockModule{BaseModule: primitives.NewBaseModule()}

	// Create test examples
	trainset := []*primitives.Example{
		primitives.NewExample(
			map[string]interface{}{"question": "What is 2+2?"},
			map[string]interface{}{"answer": "4"},
		),
		primitives.NewExample(
			map[string]interface{}{"question": "What is 3+3?"},
			map[string]interface{}{"answer": "6"},
		),
		primitives.NewExample(
			map[string]interface{}{"question": "What is 4+4?"},
			map[string]interface{}{"answer": "8"},
		),
	}

	compiled, err := optimizer.Compile(ctx, module, trainset, nil)
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}

	if compiled == nil {
		t.Fatal("Compile returned nil module")
	}

	// Module was compiled successfully
	// In a full implementation, we would check that demos were actually set
}

func TestLabeledFewShot_CompileSampling(t *testing.T) {
	ctx := context.Background()

	// Test with sampling
	optimizer1 := NewLabeledFewShot(2).WithSample(true).WithSeed(0)

	module := &mockModule{BaseModule: primitives.NewBaseModule()}
	trainset := []*primitives.Example{
		primitives.NewExample(
			map[string]interface{}{"q": "1"},
			map[string]interface{}{"a": "1"},
		),
		primitives.NewExample(
			map[string]interface{}{"q": "2"},
			map[string]interface{}{"a": "2"},
		),
		primitives.NewExample(
			map[string]interface{}{"q": "3"},
			map[string]interface{}{"a": "3"},
		),
	}

	compiled1, err := optimizer1.Compile(ctx, module, trainset, nil)
	if err != nil {
		t.Fatalf("Compile with sampling failed: %v", err)
	}

	if compiled1 == nil {
		t.Fatal("Compile returned nil module")
	}

	// Test without sampling
	optimizer2 := NewLabeledFewShot(2).WithSample(false)

	compiled2, err := optimizer2.Compile(ctx, module, trainset, nil)
	if err != nil {
		t.Fatalf("Compile without sampling failed: %v", err)
	}

	if compiled2 == nil {
		t.Fatal("Compile returned nil module")
	}
}

func TestLabeledFewShot_CompileMoreDemosThanExamples(t *testing.T) {
	ctx := context.Background()
	optimizer := NewLabeledFewShot(10) // Request more than available

	module := &mockModule{BaseModule: primitives.NewBaseModule()}
	trainset := []*primitives.Example{
		primitives.NewExample(
			map[string]interface{}{"q": "1"},
			map[string]interface{}{"a": "1"},
		),
		primitives.NewExample(
			map[string]interface{}{"q": "2"},
			map[string]interface{}{"a": "2"},
		),
	}

	compiled, err := optimizer.Compile(ctx, module, trainset, nil)
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}

	if compiled == nil {
		t.Fatal("Compile returned nil module")
	}
}
