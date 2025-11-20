// Package datasets provides dataset loading and processing utilities for DSPy.
package datasets

import (
	"context"
	"math/rand"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

// Dataset represents a generic dataset with train/dev/test splits.
type Dataset interface {
	// Train returns the training examples
	Train() []*primitives.Example

	// Dev returns the development/validation examples
	Dev() []*primitives.Example

	// Test returns the test examples
	Test() []*primitives.Example

	// Name returns the dataset name
	Name() string

	// Len returns the total number of examples across all splits
	Len() int
}

// BaseDataset provides common functionality for datasets.
type BaseDataset struct {
	name      string
	train     []*primitives.Example
	dev       []*primitives.Example
	test      []*primitives.Example
	trainSize int
	devSize   int
	testSize  int
	trainSeed int64
	evalSeed  int64
}

// DatasetOptions configures dataset loading.
type DatasetOptions struct {
	TrainSize int
	DevSize   int
	TestSize  int
	TrainSeed int64
	EvalSeed  int64
	Shuffle   bool
}

// DefaultDatasetOptions returns default dataset options.
func DefaultDatasetOptions() DatasetOptions {
	return DatasetOptions{
		TrainSize: -1, // -1 means use all
		DevSize:   -1,
		TestSize:  -1,
		TrainSeed: 0,
		EvalSeed:  2023,
		Shuffle:   false,
	}
}

// NewBaseDataset creates a new base dataset.
func NewBaseDataset(name string, opts DatasetOptions) *BaseDataset {
	return &BaseDataset{
		name:      name,
		trainSize: opts.TrainSize,
		devSize:   opts.DevSize,
		testSize:  opts.TestSize,
		trainSeed: opts.TrainSeed,
		evalSeed:  opts.EvalSeed,
	}
}

// Train implements Dataset.Train.
func (d *BaseDataset) Train() []*primitives.Example {
	return d.train
}

// Dev implements Dataset.Dev.
func (d *BaseDataset) Dev() []*primitives.Example {
	return d.dev
}

// Test implements Dataset.Test.
func (d *BaseDataset) Test() []*primitives.Example {
	return d.test
}

// Name implements Dataset.Name.
func (d *BaseDataset) Name() string {
	return d.name
}

// Len implements Dataset.Len.
func (d *BaseDataset) Len() int {
	return len(d.train) + len(d.dev) + len(d.test)
}

// SetTrain sets the training examples.
func (d *BaseDataset) SetTrain(examples []*primitives.Example) {
	d.train = d.applySizeAndShuffle(examples, d.trainSize, d.trainSeed)
}

// SetDev sets the development examples.
func (d *BaseDataset) SetDev(examples []*primitives.Example) {
	d.dev = d.applySizeAndShuffle(examples, d.devSize, d.evalSeed)
}

// SetTest sets the test examples.
func (d *BaseDataset) SetTest(examples []*primitives.Example) {
	d.test = d.applySizeAndShuffle(examples, d.testSize, d.evalSeed)
}

func (d *BaseDataset) applySizeAndShuffle(examples []*primitives.Example, size int, seed int64) []*primitives.Example {
	if len(examples) == 0 {
		return examples
	}

	// Shuffle if seed is set
	if seed != 0 {
		r := rand.New(rand.NewSource(seed))
		shuffled := make([]*primitives.Example, len(examples))
		copy(shuffled, examples)
		r.Shuffle(len(shuffled), func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})
		examples = shuffled
	}

	// Apply size limit
	if size > 0 && size < len(examples) {
		examples = examples[:size]
	}

	return examples
}

// Loader is a function that loads dataset examples.
type Loader func(ctx context.Context) ([]*primitives.Example, error)

// SplitData splits examples into train/dev/test sets.
func SplitData(examples []*primitives.Example, trainRatio, devRatio float64) (train, dev, test []*primitives.Example) {
	total := len(examples)
	trainEnd := int(float64(total) * trainRatio)
	devEnd := trainEnd + int(float64(total)*devRatio)

	if trainEnd > total {
		trainEnd = total
	}
	if devEnd > total {
		devEnd = total
	}

	train = examples[:trainEnd]
	dev = examples[trainEnd:devEnd]
	test = examples[devEnd:]

	return train, dev, test
}
