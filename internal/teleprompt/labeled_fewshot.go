package teleprompt

import (
	"context"
	"math/rand"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// LabeledFewShot is a simple optimizer that selects labeled examples for demonstrations.
// Based on dspy/teleprompt/vanilla.py
type LabeledFewShot struct {
	*BaseTeleprompt

	// K is the number of examples to select
	K int

	// Sample determines whether to randomly sample examples or take first K
	Sample bool

	// Seed for random sampling (default 0)
	Seed int64
}

// NewLabeledFewShot creates a new LabeledFewShot optimizer.
func NewLabeledFewShot(k int) *LabeledFewShot {
	return &LabeledFewShot{
		BaseTeleprompt: NewBaseTeleprompt("LabeledFewShot"),
		K:              k,
		Sample:         true,
		Seed:           0,
	}
}

// WithSample sets whether to randomly sample examples.
func (l *LabeledFewShot) WithSample(sample bool) *LabeledFewShot {
	l.Sample = sample
	return l
}

// WithSeed sets the random seed for sampling.
func (l *LabeledFewShot) WithSeed(seed int64) *LabeledFewShot {
	l.Seed = seed
	return l
}

// Compile implements Teleprompt.Compile.
// It simply assigns labeled examples as demonstrations to each predictor.
func (l *LabeledFewShot) Compile(ctx context.Context, module primitives.Module, trainset []*primitives.Example, metric interface{}) (primitives.Module, error) {
	if len(trainset) == 0 {
		// Empty trainset is valid - just return a copy of the module
		return module.Copy(), nil
	}

	// Create a copy of the student module
	student := module.Copy()

	// Create random number generator with seed
	rng := rand.New(rand.NewSource(l.Seed))

	// Get all named parameters that could be demo parameters
	params := student.NamedParameters()

	for _, namedParam := range params {
		param := namedParam.Param

		// Check if this parameter can hold demonstrations
		// In DSPy, predictors have a demos parameter
		if namedParam.Name == "demos" {
			var selectedDemos []*primitives.Example

			// Determine how many demos to select
			numDemos := l.K
			if numDemos > len(trainset) {
				numDemos = len(trainset)
			}

			if l.Sample {
				// Randomly sample K examples
				indices := rng.Perm(len(trainset))[:numDemos]
				selectedDemos = make([]*primitives.Example, numDemos)
				for i, idx := range indices {
					selectedDemos[i] = trainset[idx]
				}
			} else {
				// Take first K examples
				selectedDemos = trainset[:numDemos]
			}

			// Set the demos parameter
			param.SetValue(selectedDemos)
		}
	}

	return student, nil
}
