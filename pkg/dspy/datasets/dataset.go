// Package datasets provides dataset loading and processing utilities for DSPy.
package datasets

import (
	"context"
	"math/rand"
)

// Example represents a single example in a dataset.
type Example struct {
	ID     string
	Inputs map[string]interface{}
	Labels map[string]interface{}
}

// NewExample creates a new Example.
func NewExample(id string, inputs, labels map[string]interface{}) *Example {
	if inputs == nil {
		inputs = make(map[string]interface{})
	}
	if labels == nil {
		labels = make(map[string]interface{})
	}
	return &Example{
		ID:     id,
		Inputs: inputs,
		Labels: labels,
	}
}

// Dataset interface provides methods for loading and manipulating datasets.
type Dataset interface {
	// Load returns all examples in the dataset
	Load(ctx context.Context) ([]Example, error)
	
	// Len returns the number of examples
	Len() int
	
	// Split divides the dataset into multiple parts based on ratios
	Split(ratios ...float64) []Dataset
	
	// Batch returns examples in batches of the specified size
	Batch(size int) [][]Example
	
	// Shuffle randomly shuffles the dataset
	Shuffle(seed int64) Dataset
	
	// Filter returns a new dataset with examples that match the predicate
	Filter(predicate func(Example) bool) Dataset
}

// InMemoryDataset is a simple in-memory implementation of Dataset.
type InMemoryDataset struct {
	examples []Example
	name     string
}

// NewInMemoryDataset creates a new in-memory dataset.
func NewInMemoryDataset(name string, examples []Example) *InMemoryDataset {
	return &InMemoryDataset{
		name:     name,
		examples: examples,
	}
}

// Load implements Dataset.Load.
func (d *InMemoryDataset) Load(ctx context.Context) ([]Example, error) {
	return d.examples, nil
}

// Len implements Dataset.Len.
func (d *InMemoryDataset) Len() int {
	return len(d.examples)
}

// Split implements Dataset.Split.
func (d *InMemoryDataset) Split(ratios ...float64) []Dataset {
	if len(ratios) == 0 {
		return []Dataset{d}
	}
	
	// Normalize ratios
	total := 0.0
	for _, r := range ratios {
		total += r
	}
	
	datasets := make([]Dataset, len(ratios))
	start := 0
	
	for i, ratio := range ratios {
		size := int(float64(len(d.examples)) * ratio / total)
		if i == len(ratios)-1 {
			// Last split gets remaining examples
			size = len(d.examples) - start
		}
		
		end := start + size
		if end > len(d.examples) {
			end = len(d.examples)
		}
		
		datasets[i] = NewInMemoryDataset(d.name, d.examples[start:end])
		start = end
	}
	
	return datasets
}

// Batch implements Dataset.Batch.
func (d *InMemoryDataset) Batch(size int) [][]Example {
	if size <= 0 {
		size = 1
	}
	
	batches := make([][]Example, 0, (len(d.examples)+size-1)/size)
	for i := 0; i < len(d.examples); i += size {
		end := i + size
		if end > len(d.examples) {
			end = len(d.examples)
		}
		batches = append(batches, d.examples[i:end])
	}
	
	return batches
}

// Shuffle implements Dataset.Shuffle.
func (d *InMemoryDataset) Shuffle(seed int64) Dataset {
	shuffled := make([]Example, len(d.examples))
	copy(shuffled, d.examples)
	
	r := rand.New(rand.NewSource(seed))
	r.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	
	return NewInMemoryDataset(d.name, shuffled)
}

// Filter implements Dataset.Filter.
func (d *InMemoryDataset) Filter(predicate func(Example) bool) Dataset {
	filtered := make([]Example, 0)
	for _, ex := range d.examples {
		if predicate(ex) {
			filtered = append(filtered, ex)
		}
	}
	
	return NewInMemoryDataset(d.name, filtered)
}

// Name returns the dataset name.
func (d *InMemoryDataset) Name() string {
	return d.name
}
