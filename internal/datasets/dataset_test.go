package datasets

import (
	"os"
	"path/filepath"
	"testing"
)

func TestInMemoryDataset(t *testing.T) {
	examples := []map[string]interface{}{
		{"question": "What is 2+2?", "answer": "4"},
		{"question": "What is 3+3?", "answer": "6"},
	}

	dataset := NewInMemoryDataset(examples)

	if dataset.Len() != 2 {
		t.Errorf("Expected length 2, got %d", dataset.Len())
	}

	ex, err := dataset.Get(0)
	if err != nil {
		t.Fatalf("Get(0) error = %v", err)
	}
	if ex["question"] != "What is 2+2?" {
		t.Errorf("Expected question 'What is 2+2?', got '%v'", ex["question"])
	}

	_, err = dataset.Get(5)
	if err == nil {
		t.Error("Expected error for out of range index, got nil")
	}

	all, err := dataset.GetAll()
	if err != nil {
		t.Fatalf("GetAll() error = %v", err)
	}
	if len(all) != 2 {
		t.Errorf("Expected 2 examples, got %d", len(all))
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

	// Test loading all fields
	dataset, err := LoadFromJSON(jsonPath, nil)
	if err != nil {
		t.Fatalf("LoadFromJSON() error = %v", err)
	}

	if dataset.Len() != 2 {
		t.Errorf("Expected 2 examples, got %d", dataset.Len())
	}

	// Test loading specific fields
	dataset, err = LoadFromJSON(jsonPath, []string{"question", "answer"})
	if err != nil {
		t.Fatalf("LoadFromJSON() with fields error = %v", err)
	}

	ex, _ := dataset.Get(0)
	if _, hasExtra := ex["extra"]; hasExtra {
		t.Error("Expected 'extra' field to be filtered out")
	}
	if ex["question"] != "Q1" {
		t.Errorf("Expected question 'Q1', got '%v'", ex["question"])
	}
}

func TestLoadFromCSV(t *testing.T) {
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "test.csv")

	csvData := `question,answer,extra
Q1,A1,E1
Q2,A2,E2`

	if err := os.WriteFile(csvPath, []byte(csvData), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	// Test loading all fields
	dataset, err := LoadFromCSV(csvPath, nil)
	if err != nil {
		t.Fatalf("LoadFromCSV() error = %v", err)
	}

	if dataset.Len() != 2 {
		t.Errorf("Expected 2 examples, got %d", dataset.Len())
	}

	ex, _ := dataset.Get(0)
	if ex["question"] != "Q1" {
		t.Errorf("Expected question 'Q1', got '%v'", ex["question"])
	}

	// Test loading specific fields
	dataset, err = LoadFromCSV(csvPath, []string{"question"})
	if err != nil {
		t.Fatalf("LoadFromCSV() with fields error = %v", err)
	}

	ex, _ = dataset.Get(0)
	if _, hasAnswer := ex["answer"]; hasAnswer {
		t.Error("Expected 'answer' field to be filtered out")
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

	dataset, err := LoadFromJSONL(jsonlPath, nil)
	if err != nil {
		t.Fatalf("LoadFromJSONL() error = %v", err)
	}

	if dataset.Len() != 2 {
		t.Errorf("Expected 2 examples, got %d", dataset.Len())
	}

	ex, _ := dataset.Get(1)
	if ex["question"] != "Q2" {
		t.Errorf("Expected question 'Q2', got '%v'", ex["question"])
	}
}
