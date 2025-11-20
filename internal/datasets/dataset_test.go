package datasets

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

func TestBaseDataset(t *testing.T) {
	opts := DefaultDatasetOptions()
	dataset := NewBaseDataset("test", opts)

	// Create some test examples
	examples := []*primitives.Example{
		primitives.NewExample(map[string]interface{}{"question": "What is 2+2?"}, map[string]interface{}{"answer": "4"}),
		primitives.NewExample(map[string]interface{}{"question": "What is 3+3?"}, map[string]interface{}{"answer": "6"}),
	}

	dataset.SetTrain(examples)

	if dataset.Name() != "test" {
		t.Errorf("Expected name 'test', got '%s'", dataset.Name())
	}

	if len(dataset.Train()) != 2 {
		t.Errorf("Expected 2 training examples, got %d", len(dataset.Train()))
	}

	if dataset.Len() != 2 {
		t.Errorf("Expected total length 2, got %d", dataset.Len())
	}
}

func TestDatasetSplits(t *testing.T) {
	examples := []*primitives.Example{
		primitives.NewExample(nil, map[string]interface{}{"id": 1}),
		primitives.NewExample(nil, map[string]interface{}{"id": 2}),
		primitives.NewExample(nil, map[string]interface{}{"id": 3}),
		primitives.NewExample(nil, map[string]interface{}{"id": 4}),
		primitives.NewExample(nil, map[string]interface{}{"id": 5}),
		primitives.NewExample(nil, map[string]interface{}{"id": 6}),
		primitives.NewExample(nil, map[string]interface{}{"id": 7}),
		primitives.NewExample(nil, map[string]interface{}{"id": 8}),
		primitives.NewExample(nil, map[string]interface{}{"id": 9}),
		primitives.NewExample(nil, map[string]interface{}{"id": 10}),
	}

	train, dev, test := SplitData(examples, 0.6, 0.2)

	if len(train) != 6 {
		t.Errorf("Expected 6 training examples, got %d", len(train))
	}
	if len(dev) != 2 {
		t.Errorf("Expected 2 dev examples, got %d", len(dev))
	}
	if len(test) != 2 {
		t.Errorf("Expected 2 test examples, got %d", len(test))
	}
}

func TestLoadFromJSON(t *testing.T) {
	// Create a temporary JSON file
	tmpDir := t.TempDir()
	jsonPath := filepath.Join(tmpDir, "test.json")

	jsonData := `[
		{"question": "Q1", "answer": "A1", "extra": "E1"},
		{"question": "Q2", "answer": "A2", "extra": "E2"}
	]`

	if err := os.WriteFile(jsonPath, []byte(jsonData), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	examples, err := LoadFromJSON(jsonPath)
	if err != nil {
		t.Fatalf("LoadFromJSON() error = %v", err)
	}

	if len(examples) != 2 {
		t.Errorf("Expected 2 examples, got %d", len(examples))
	}

	// Check first example
	ex := examples[0]
	if val, ok := ex.Get("question"); !ok || val != "Q1" {
		t.Errorf("Expected question 'Q1', got '%v'", val)
	}
	if val, ok := ex.Get("answer"); !ok || val != "A1" {
		t.Errorf("Expected answer 'A1', got '%v'", val)
	}
}

func TestLoadFromJSONL(t *testing.T) {
	tmpDir := t.TempDir()
	jsonlPath := filepath.Join(tmpDir, "test.jsonl")

	jsonlData := `{"question": "Q1", "answer": "A1"}
{"question": "Q2", "answer": "A2"}`

	if err := os.WriteFile(jsonlPath, []byte(jsonlData), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	examples, err := LoadFromJSONL(jsonlPath)
	if err != nil {
		t.Fatalf("LoadFromJSONL() error = %v", err)
	}

	if len(examples) != 2 {
		t.Errorf("Expected 2 examples, got %d", len(examples))
	}

	ex := examples[1]
	if val, ok := ex.Get("question"); !ok || val != "Q2" {
		t.Errorf("Expected question 'Q2', got '%v'", val)
	}
}

func TestDataLoader(t *testing.T) {
	examples := []*primitives.Example{
		primitives.NewExample(nil, map[string]interface{}{"id": 1}),
		primitives.NewExample(nil, map[string]interface{}{"id": 2}),
		primitives.NewExample(nil, map[string]interface{}{"id": 3}),
		primitives.NewExample(nil, map[string]interface{}{"id": 4}),
		primitives.NewExample(nil, map[string]interface{}{"id": 5}),
	}

	opts := DataLoaderOptions{
		BatchSize:  2,
		Shuffle:    false,
		Seed:       0,
		NumWorkers: 1,
		DropLast:   false,
	}

	dl := NewDataLoader(examples, opts)

	if dl.Len() != 5 {
		t.Errorf("Expected length 5, got %d", dl.Len())
	}

	if dl.NumBatches() != 3 {
		t.Errorf("Expected 3 batches, got %d", dl.NumBatches())
	}

	// Test iteration
	batchCount := 0
	totalExamples := 0
	for {
		batch := dl.Next()
		if batch == nil {
			break
		}
		batchCount++
		totalExamples += len(batch)
	}

	if batchCount != 3 {
		t.Errorf("Expected 3 batches, got %d", batchCount)
	}
	if totalExamples != 5 {
		t.Errorf("Expected 5 total examples, got %d", totalExamples)
	}

	// Test reset
	dl.Reset()
	batch := dl.Next()
	if batch == nil {
		t.Error("Expected batch after reset, got nil")
	}
}
