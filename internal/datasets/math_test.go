package datasets

import (
	"strings"
	"testing"

	"github.com/stanfordnlp/dspy/internal/primitives"
)

func TestExtractAnswer(t *testing.T) {
	tests := []struct {
		name     string
		solution string
		expected string
	}{
		{
			name:     "Simple boxed answer",
			solution: "The answer is \\boxed{42}",
			expected: "42",
		},
		{
			name:     "Boxed with nested braces",
			solution: "The answer is \\boxed{\\frac{1}{2}}",
			expected: "\\frac{1}{2}",
		},
		{
			name:     "Boxed with text command",
			solution: "The answer is \\boxed{42 \\text{ meters}}",
			expected: "42",
		},
		{
			name:     "No boxed command",
			solution: "The answer is 42",
			expected: "",
		},
		{
			name:     "Complex nested braces",
			solution: "The answer is \\boxed{\\left[\\frac{1}{2}, \\frac{4}{3}\\right]}",
			expected: "\\left[\\frac{1}{2}, \\frac{4}{3}\\right]",
		},
		{
			name:     "With exclamation marks",
			solution: "The answer is \\boxed{42\\!}",
			expected: "42",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ExtractAnswer(tt.solution)
			if result != tt.expected {
				t.Errorf("ExtractAnswer() = %q, expected %q", result, tt.expected)
			}
		})
	}
}

func TestProcessMATHExample(t *testing.T) {
	raw := MATHExample{
		Problem:  "What is 2+2?",
		Solution: "The answer is \\boxed{4}",
		Level:    "1",
		Type:     "algebra",
	}

	ex := ProcessMATHExample(raw)

	// Verify fields
	if question, ok := ex.Get("question"); !ok || question != "What is 2+2?" {
		t.Errorf("Expected question 'What is 2+2?', got %v", question)
	}

	if answer, ok := ex.Get("answer"); !ok || answer != "4" {
		t.Errorf("Expected answer '4', got %v", answer)
	}

	if level, ok := ex.Get("level"); !ok || level != "1" {
		t.Errorf("Expected level '1', got %v", level)
	}

	if typ, ok := ex.Get("type"); !ok || typ != "algebra" {
		t.Errorf("Expected type 'algebra', got %v", typ)
	}

	// Verify question is marked as input
	if _, ok := ex.Inputs()["question"]; !ok {
		t.Error("Expected 'question' to be in inputs")
	}
}

func TestLoadMATHFromJSON(t *testing.T) {
	jsonData := `[
		{
			"problem": "What is 2+2?",
			"solution": "The answer is \\boxed{4}",
			"level": "1",
			"type": "algebra"
		},
		{
			"problem": "What is 3+3?",
			"solution": "The answer is \\boxed{6}",
			"level": "2",
			"type": "arithmetic"
		}
	]`

	reader := strings.NewReader(jsonData)
	examples, err := LoadMATHFromJSON(reader)
	if err != nil {
		t.Fatalf("LoadMATHFromJSON() error = %v", err)
	}

	if len(examples) != 2 {
		t.Errorf("Expected 2 examples, got %d", len(examples))
	}

	// Check first example
	if question, ok := examples[0].Get("question"); !ok || question != "What is 2+2?" {
		t.Errorf("Expected question 'What is 2+2?', got %v", question)
	}

	if answer, ok := examples[0].Get("answer"); !ok || answer != "4" {
		t.Errorf("Expected answer '4', got %v", answer)
	}
}

func TestCreateMATHDatasetFromExamples(t *testing.T) {
	// Create test examples
	examples := make([]*primitives.Example, 30)
	for i := 0; i < 30; i++ {
		examples[i] = primitives.NewExample(nil, map[string]interface{}{
			"question": "Question",
			"answer":   "42",
		})
	}

	opts := DefaultMATHOptions()
	dataset := CreateMATHDatasetFromExamples(examples, opts)

	if dataset.Name() != "math" {
		t.Errorf("Expected name 'math', got '%s'", dataset.Name())
	}

	// Verify splits exist
	if len(dataset.Train()) == 0 {
		t.Error("Expected non-empty training set")
	}
	if len(dataset.Dev()) == 0 {
		t.Error("Expected non-empty dev set")
	}
	if len(dataset.Test()) == 0 {
		t.Error("Expected non-empty test set")
	}

	// Verify total equals original
	total := len(dataset.Train()) + len(dataset.Dev()) + len(dataset.Test())
	if total > len(examples) {
		t.Errorf("Total split size (%d) exceeds original size (%d)", total, len(examples))
	}
}

func TestMATHMetric(t *testing.T) {
	tests := []struct {
		name     string
		gold     *primitives.Example
		pred     *primitives.Example
		expected bool
	}{
		{
			name:     "Exact match",
			gold:     primitives.NewExample(nil, map[string]interface{}{"answer": "42"}),
			pred:     primitives.NewExample(nil, map[string]interface{}{"answer": "42"}),
			expected: true,
		},
		{
			name:     "Mismatch",
			gold:     primitives.NewExample(nil, map[string]interface{}{"answer": "42"}),
			pred:     primitives.NewExample(nil, map[string]interface{}{"answer": "43"}),
			expected: false,
		},
		{
			name:     "Whitespace trimmed",
			gold:     primitives.NewExample(nil, map[string]interface{}{"answer": "42"}),
			pred:     primitives.NewExample(nil, map[string]interface{}{"answer": "  42  "}),
			expected: true,
		},
		{
			name:     "Missing gold answer",
			gold:     primitives.NewExample(nil, map[string]interface{}{}),
			pred:     primitives.NewExample(nil, map[string]interface{}{"answer": "42"}),
			expected: false,
		},
		{
			name:     "Missing pred answer",
			gold:     primitives.NewExample(nil, map[string]interface{}{"answer": "42"}),
			pred:     primitives.NewExample(nil, map[string]interface{}{}),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MATHMetric(tt.gold, tt.pred)
			if result != tt.expected {
				t.Errorf("MATHMetric() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestMATHExampleFields(t *testing.T) {
	raw := MATHExample{
		Problem:  "Solve for x: 2x = 4",
		Solution: "Divide both sides by 2 to get x = \\boxed{2}",
		Level:    "2",
		Type:     "algebra",
	}

	ex := ProcessMATHExample(raw)

	// Verify all fields are present
	fields := []string{"question", "reasoning", "answer", "level", "type"}
	for _, field := range fields {
		if _, ok := ex.Get(field); !ok {
			t.Errorf("Expected field %s to be present", field)
		}
	}
}
