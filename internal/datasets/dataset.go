// Package datasets provides dataset loading and management for DSPy.
package datasets

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// Dataset represents a collection of examples for training or evaluation.
type Dataset interface {
	// Len returns the number of examples in the dataset.
	Len() int
	// Get returns the example at the given index.
	Get(index int) (map[string]interface{}, error)
	// GetAll returns all examples in the dataset.
	GetAll() ([]map[string]interface{}, error)
}

// InMemoryDataset is a simple in-memory dataset implementation.
type InMemoryDataset struct {
	examples []map[string]interface{}
}

// NewInMemoryDataset creates a new in-memory dataset.
func NewInMemoryDataset(examples []map[string]interface{}) *InMemoryDataset {
	return &InMemoryDataset{
		examples: examples,
	}
}

// Len returns the number of examples in the dataset.
func (d *InMemoryDataset) Len() int {
	return len(d.examples)
}

// Get returns the example at the given index.
func (d *InMemoryDataset) Get(index int) (map[string]interface{}, error) {
	if index < 0 || index >= len(d.examples) {
		return nil, fmt.Errorf("index %d out of range [0, %d)", index, len(d.examples))
	}
	return d.examples[index], nil
}

// GetAll returns all examples in the dataset.
func (d *InMemoryDataset) GetAll() ([]map[string]interface{}, error) {
	return d.examples, nil
}

// LoadFromJSON loads a dataset from a JSON file.
// The JSON file should contain an array of objects, where each object represents an example.
func LoadFromJSON(path string, fields []string) (Dataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening file: %w", err)
	}
	defer file.Close()

	var data []map[string]interface{}
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return nil, fmt.Errorf("decoding JSON: %w", err)
	}

	// Filter fields if specified
	if len(fields) > 0 {
		filtered := make([]map[string]interface{}, len(data))
		for i, example := range data {
			filtered[i] = make(map[string]interface{})
			for _, field := range fields {
				if val, ok := example[field]; ok {
					filtered[i][field] = val
				}
			}
		}
		data = filtered
	}

	return NewInMemoryDataset(data), nil
}

// LoadFromCSV loads a dataset from a CSV file.
// The first row is expected to contain column headers.
func LoadFromCSV(path string, fields []string) (Dataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Read header
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("reading header: %w", err)
	}

	// Determine which fields to include
	fieldIndices := make(map[string]int)
	if len(fields) == 0 {
		// Include all fields
		for i, name := range header {
			fieldIndices[name] = i
		}
	} else {
		// Include only specified fields
		for _, field := range fields {
			for i, name := range header {
				if name == field {
					fieldIndices[field] = i
					break
				}
			}
		}
	}

	// Read data
	var examples []map[string]interface{}
	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading row: %w", err)
		}

		example := make(map[string]interface{})
		for name, idx := range fieldIndices {
			if idx < len(row) {
				example[name] = row[idx]
			}
		}
		examples = append(examples, example)
	}

	return NewInMemoryDataset(examples), nil
}

// LoadFromJSONL loads a dataset from a JSONL (JSON Lines) file.
// Each line in the file should be a valid JSON object.
func LoadFromJSONL(path string, fields []string) (Dataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening file: %w", err)
	}
	defer file.Close()

	var examples []map[string]interface{}
	decoder := json.NewDecoder(file)

	for {
		var example map[string]interface{}
		if err := decoder.Decode(&example); err == io.EOF {
			break
		} else if err != nil {
			return nil, fmt.Errorf("decoding JSONL line: %w", err)
		}

		// Filter fields if specified
		if len(fields) > 0 {
			filtered := make(map[string]interface{})
			for _, field := range fields {
				if val, ok := example[field]; ok {
					filtered[field] = val
				}
			}
			example = filtered
		}

		examples = append(examples, example)
	}

	return NewInMemoryDataset(examples), nil
}
