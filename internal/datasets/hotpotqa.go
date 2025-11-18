package datasets

import (
	"encoding/json"
	"fmt"
	"os"
)

// HotPotQAExample represents a single HotPotQA example.
type HotPotQAExample struct {
	Question      string              `json:"question"`
	Answer        string              `json:"answer"`
	Context       [][]string          `json:"context"`
	Type          string              `json:"type"`
	Level         string              `json:"level"`
	SupportingFacts [][]interface{}  `json:"supporting_facts"`
	ID            string              `json:"_id"`
}

// HotPotQADataset is a dataset for the HotPotQA benchmark.
type HotPotQADataset struct {
	examples []map[string]interface{}
}

// LoadHotPotQA loads the HotPotQA dataset from a JSON file.
func LoadHotPotQA(path string, includeContext bool) (Dataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening file: %w", err)
	}
	defer file.Close()

	var rawExamples []HotPotQAExample
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&rawExamples); err != nil {
		return nil, fmt.Errorf("decoding JSON: %w", err)
	}

	examples := make([]map[string]interface{}, len(rawExamples))
	for i, ex := range rawExamples {
		example := map[string]interface{}{
			"question": ex.Question,
			"answer":   ex.Answer,
			"type":     ex.Type,
			"level":    ex.Level,
			"id":       ex.ID,
		}

		if includeContext {
			// Flatten context into a single string or list of passages
			var passages []string
			for _, ctx := range ex.Context {
				if len(ctx) > 1 {
					passages = append(passages, ctx[1]) // ctx[1] is the text
				}
			}
			example["context"] = passages
		}

		examples[i] = example
	}

	return NewInMemoryDataset(examples), nil
}

// Len returns the number of examples in the dataset.
func (d *HotPotQADataset) Len() int {
	return len(d.examples)
}

// Get returns the example at the given index.
func (d *HotPotQADataset) Get(index int) (map[string]interface{}, error) {
	if index < 0 || index >= len(d.examples) {
		return nil, fmt.Errorf("index %d out of range [0, %d)", index, len(d.examples))
	}
	return d.examples[index], nil
}

// GetAll returns all examples in the dataset.
func (d *HotPotQADataset) GetAll() ([]map[string]interface{}, error) {
	return d.examples, nil
}
