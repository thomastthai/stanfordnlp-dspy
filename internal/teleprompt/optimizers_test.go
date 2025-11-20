package teleprompt

import (
	"context"
	"testing"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// Helper function to create test examples
func createTestExamples(n int) []*primitives.Example {
	examples := make([]*primitives.Example, n)
	for i := 0; i < n; i++ {
		examples[i] = primitives.NewExample(
			map[string]interface{}{"input": i},
			map[string]interface{}{"output": i * 2},
		)
	}
	return examples
}

// Helper function to create a test module
func createTestModule() primitives.Module {
	return &mockModule{
		BaseModule: primitives.NewBaseModule(),
	}
}

// Test all optimizers can be instantiated
func TestOptimizers_Instantiation(t *testing.T) {
	tests := []struct {
		name     string
		create   func() Teleprompt
		wantName string
	}{
		{
			name:     "LabeledFewShot",
			create:   func() Teleprompt { return NewLabeledFewShot(5) },
			wantName: "LabeledFewShot",
		},
		{
			name:     "KNNFewShot",
			create:   func() Teleprompt { return NewKNNFewShot(3, nil, nil) },
			wantName: "KNNFewShot",
		},
		{
			name:     "Ensemble",
			create:   func() Teleprompt { return NewEnsemble() },
			wantName: "Ensemble",
		},
		{
			name:     "RandomSearch",
			create:   func() Teleprompt { return NewRandomSearch(nil) },
			wantName: "RandomSearch",
		},
		{
			name:     "BootstrapFinetune",
			create:   func() Teleprompt { return NewBootstrapFinetune(nil) },
			wantName: "BootstrapFinetune",
		},
		{
			name:     "COPRO",
			create:   func() Teleprompt { return NewCOPRO(func() {}) },
			wantName: "COPRO",
		},
		{
			name:     "MIPROv2",
			create:   func() Teleprompt { return NewMIPROv2(nil) },
			wantName: "MIPROv2",
		},
		{
			name:     "AvatarOptimizer",
			create:   func() Teleprompt { return NewAvatarOptimizer(nil) },
			wantName: "AvatarOptimizer",
		},
		{
			name:     "SIMBA",
			create:   func() Teleprompt { return NewSIMBA(nil) },
			wantName: "SIMBA",
		},
		{
			name:     "BetterTogether",
			create:   func() Teleprompt { return NewBetterTogether(nil) },
			wantName: "BetterTogether",
		},
		{
			name:     "InferRules",
			create:   func() Teleprompt { return NewInferRules(nil) },
			wantName: "BootstrapFewShot", // InferRules extends BootstrapFewShot
		},
		{
			name:     "OptunaOptimizer",
			create:   func() Teleprompt { return NewOptunaOptimizer(nil) },
			wantName: "OptunaOptimizer",
		},
		{
			name:     "BootstrapTrace",
			create:   func() Teleprompt { return NewBootstrapTrace(4) },
			wantName: "BootstrapTrace",
		},
		{
			name:     "GRPO",
			create:   func() Teleprompt { return NewGRPO(nil) },
			wantName: "GRPO",
		},
		{
			name:     "SignatureOptimizer",
			create:   func() Teleprompt { return NewSignatureOptimizer(nil) },
			wantName: "SignatureOptimizer",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			optimizer := tt.create()
			if optimizer == nil {
				t.Fatalf("%s: constructor returned nil", tt.name)
			}

			if optimizer.Name() != tt.wantName {
				t.Errorf("%s: expected name %q, got %q", tt.name, tt.wantName, optimizer.Name())
			}
		})
	}
}

// Test that optimizers implement the Teleprompt interface
func TestOptimizers_InterfaceCompliance(t *testing.T) {
	ctx := context.Background()
	module := createTestModule()
	trainset := createTestExamples(5)

	optimizers := []Teleprompt{
		NewLabeledFewShot(3),
		NewEnsemble(),
		NewRandomSearch(nil),
		NewBootstrapFinetune(nil),
		NewMIPROv2(nil),
		NewAvatarOptimizer(nil),
		NewSIMBA(nil),
		NewBetterTogether(nil),
		NewOptunaOptimizer(nil),
		NewBootstrapTrace(4),
		NewGRPO(nil),
		NewSignatureOptimizer(nil),
	}

	for _, optimizer := range optimizers {
		t.Run(optimizer.Name(), func(t *testing.T) {
			// Test that Compile doesn't panic
			_, err := optimizer.Compile(ctx, module, trainset, nil)
			// Some optimizers might return errors for missing required params, which is fine
			if err != nil {
				t.Logf("%s returned error (expected for incomplete setup): %v", optimizer.Name(), err)
			}
		})
	}
}

// Test RandomSearch with different configurations
func TestRandomSearch_Configurations(t *testing.T) {
	ctx := context.Background()
	module := createTestModule()
	trainset := createTestExamples(10)

	tests := []struct {
		name          string
		numCandidates int
		wantError     bool
	}{
		{
			name:          "few candidates",
			numCandidates: 3,
			wantError:     true, // Needs metric
		},
		{
			name:          "many candidates",
			numCandidates: 10,
			wantError:     true, // Needs metric
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			optimizer := NewRandomSearch(nil).
				WithNumCandidatePrograms(tt.numCandidates)

			_, err := optimizer.Compile(ctx, module, trainset, nil)
			if (err != nil) != tt.wantError {
				t.Errorf("expected error: %v, got: %v", tt.wantError, err)
			}
		})
	}
}

// Test MIPROv2 auto modes
func TestMIPROv2_AutoModes(t *testing.T) {
	tests := []struct {
		mode           string
		wantCandidates int
	}{
		{"light", 6},
		{"medium", 12},
		{"heavy", 18},
	}

	for _, tt := range tests {
		t.Run(tt.mode, func(t *testing.T) {
			optimizer := NewMIPROv2(nil).WithAuto(tt.mode)

			// Check that the mode is set
			if optimizer.Auto != tt.mode {
				t.Errorf("expected auto mode %q, got %q", tt.mode, optimizer.Auto)
			}
		})
	}
}

// Test SIMBA bandit algorithms
func TestSIMBA_BanditAlgorithms(t *testing.T) {
	algorithms := []string{"ucb", "thompson", "epsilon_greedy"}

	for _, algo := range algorithms {
		t.Run(algo, func(t *testing.T) {
			optimizer := NewSIMBA(nil).WithBanditAlgorithm(algo)

			if optimizer.BanditAlgorithm != algo {
				t.Errorf("expected algorithm %q, got %q", algo, optimizer.BanditAlgorithm)
			}
		})
	}
}

// Test BetterTogether collaboration strategies
func TestBetterTogether_CollaborationStrategies(t *testing.T) {
	strategies := []string{"sequential", "parallel", "hierarchical"}

	for _, strategy := range strategies {
		t.Run(strategy, func(t *testing.T) {
			optimizer := NewBetterTogether(nil).WithCollaborationStrategy(strategy)

			if optimizer.CollaborationStrategy != strategy {
				t.Errorf("expected strategy %q, got %q", strategy, optimizer.CollaborationStrategy)
			}
		})
	}
}

// Test OptunaOptimizer samplers
func TestOptunaOptimizer_Samplers(t *testing.T) {
	samplers := []string{"tpe", "random", "grid"}

	for _, sampler := range samplers {
		t.Run(sampler, func(t *testing.T) {
			optimizer := NewOptunaOptimizer(nil).WithSampler(sampler)

			if optimizer.Sampler != sampler {
				t.Errorf("expected sampler %q, got %q", sampler, optimizer.Sampler)
			}
		})
	}
}

// Test BootstrapFinetune configuration
func TestBootstrapFinetune_Configuration(t *testing.T) {
	optimizer := NewBootstrapFinetune(nil).
		WithMaxBootstrappedDemos(20).
		WithMetricThreshold(0.8)

	if optimizer.MaxBootstrappedDemos != 20 {
		t.Errorf("expected MaxBootstrappedDemos=20, got %d", optimizer.MaxBootstrappedDemos)
	}

	if optimizer.MetricThreshold == nil || *optimizer.MetricThreshold != 0.8 {
		t.Error("expected MetricThreshold=0.8")
	}
}

// Test COPRO configuration
func TestCOPRO_Configuration(t *testing.T) {
	optimizer := NewCOPRO(func() {}).
		WithBreadth(15).
		WithDepth(5).
		WithInitTemperature(1.5)

	if optimizer.Breadth != 15 {
		t.Errorf("expected Breadth=15, got %d", optimizer.Breadth)
	}

	if optimizer.Depth != 5 {
		t.Errorf("expected Depth=5, got %d", optimizer.Depth)
	}

	if optimizer.InitTemperature != 1.5 {
		t.Errorf("expected InitTemperature=1.5, got %f", optimizer.InitTemperature)
	}
}

// Test AvatarOptimizer configuration
func TestAvatarOptimizer_Configuration(t *testing.T) {
	optimizer := NewAvatarOptimizer(nil).
		WithNumActors(5).
		WithNumIterations(20).
		WithVerbose(true)

	if optimizer.NumActors != 5 {
		t.Errorf("expected NumActors=5, got %d", optimizer.NumActors)
	}

	if optimizer.NumIterations != 20 {
		t.Errorf("expected NumIterations=20, got %d", optimizer.NumIterations)
	}

	if !optimizer.Verbose {
		t.Error("expected Verbose=true")
	}
}

// Test BootstrapTrace configuration
func TestBootstrapTrace_Configuration(t *testing.T) {
	optimizer := NewBootstrapTrace(10).
		WithMaxLabeledDemos(20).
		WithTraceMode("minimal").
		WithNumThreads(4)

	if optimizer.MaxBootstrappedDemos != 10 {
		t.Errorf("expected MaxBootstrappedDemos=10, got %d", optimizer.MaxBootstrappedDemos)
	}

	if optimizer.MaxLabeledDemos != 20 {
		t.Errorf("expected MaxLabeledDemos=20, got %d", optimizer.MaxLabeledDemos)
	}

	if optimizer.TraceMode != "minimal" {
		t.Errorf("expected TraceMode='minimal', got '%s'", optimizer.TraceMode)
	}

	if optimizer.NumThreads != 4 {
		t.Errorf("expected NumThreads=4, got %d", optimizer.NumThreads)
	}
}

// Test GRPO configuration
func TestGRPO_Configuration(t *testing.T) {
	optimizer := NewGRPO(nil).
		WithNumEpochs(5).
		WithBatchSize(64).
		WithLearningRate(1e-3).
		WithGamma(0.95).
		WithClipRange(0.3)

	if optimizer.NumEpochs != 5 {
		t.Errorf("expected NumEpochs=5, got %d", optimizer.NumEpochs)
	}

	if optimizer.BatchSize != 64 {
		t.Errorf("expected BatchSize=64, got %d", optimizer.BatchSize)
	}

	if optimizer.LearningRate != 1e-3 {
		t.Errorf("expected LearningRate=1e-3, got %f", optimizer.LearningRate)
	}

	if optimizer.Gamma != 0.95 {
		t.Errorf("expected Gamma=0.95, got %f", optimizer.Gamma)
	}

	if optimizer.ClipRange != 0.3 {
		t.Errorf("expected ClipRange=0.3, got %f", optimizer.ClipRange)
	}
}

// Test SignatureOptimizer configuration
func TestSignatureOptimizer_Configuration(t *testing.T) {
	optimizer := NewSignatureOptimizer(func() {}).
		WithOptimizeInstructions(false).
		WithOptimizeDescriptions(true).
		WithNumCandidates(15).
		WithBreadth(12).
		WithDepth(4)

	if optimizer.OptimizeInstructions {
		t.Error("expected OptimizeInstructions=false")
	}

	if !optimizer.OptimizeDescriptions {
		t.Error("expected OptimizeDescriptions=true")
	}

	if optimizer.NumCandidates != 15 {
		t.Errorf("expected NumCandidates=15, got %d", optimizer.NumCandidates)
	}

	if optimizer.Breadth != 12 {
		t.Errorf("expected Breadth=12, got %d", optimizer.Breadth)
	}

	if optimizer.Depth != 4 {
		t.Errorf("expected Depth=4, got %d", optimizer.Depth)
	}
}
