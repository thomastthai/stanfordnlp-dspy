package adapters

import (
	"strings"
	"testing"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

func TestBAMLAdapter_Format(t *testing.T) {
	adapter := NewBAMLAdapter()

	sig, err := signatures.NewSignature("question -> answer")
	if err != nil {
		t.Fatalf("failed to create signature: %v", err)
	}
	sig.Instructions = "Answer the question concisely."

	inputs := map[string]interface{}{
		"question": "What is Go?",
	}

	req, err := adapter.Format(sig, inputs, nil)
	if err != nil {
		t.Fatalf("failed to format: %v", err)
	}

	// Check for system message
	if len(req.Messages) < 2 {
		t.Errorf("expected at least 2 messages (system + user), got %d", len(req.Messages))
	}

	if req.Messages[0].Role != "system" {
		t.Errorf("expected first message to be system, got %s", req.Messages[0].Role)
	}

	// System message should contain BAML-style formatting
	systemContent := req.Messages[0].Content
	if !strings.Contains(systemContent, "Your input fields are") {
		t.Error("system message should contain field descriptions")
	}

	if !strings.Contains(systemContent, "Your output fields are") {
		t.Error("system message should contain output field descriptions")
	}

	// Check for response_format config
	if format, ok := req.Config["response_format"]; ok {
		if formatMap, ok := format.(map[string]string); ok {
			if formatMap["type"] != "json_object" {
				t.Error("response_format should be json_object")
			}
		}
	}
}

func TestBAMLAdapter_FormatWithDemos(t *testing.T) {
	adapter := NewBAMLAdapter()

	sig, err := signatures.NewSignature("question -> answer")
	if err != nil {
		t.Fatalf("failed to create signature: %v", err)
	}

	demos := []map[string]interface{}{
		{
			"question": "What is Python?",
			"answer":   "A programming language",
		},
	}

	inputs := map[string]interface{}{
		"question": "What is Go?",
	}

	req, err := adapter.Format(sig, inputs, demos)
	if err != nil {
		t.Fatalf("failed to format: %v", err)
	}

	// Should have: system + demo user + demo assistant + current user
	if len(req.Messages) < 4 {
		t.Errorf("expected at least 4 messages with demos, got %d", len(req.Messages))
	}

	// Check demo messages contain BAML-style markers
	foundMarker := false
	for _, msg := range req.Messages {
		if strings.Contains(msg.Content, "[[") && strings.Contains(msg.Content, "]]") {
			foundMarker = true
			break
		}
	}

	if !foundMarker {
		t.Error("expected to find BAML-style markers [[ ]] in messages")
	}
}

func TestBAMLAdapter_Parse(t *testing.T) {
	adapter := NewBAMLAdapter()

	sig, err := signatures.NewSignature("question -> answer")
	if err != nil {
		t.Fatalf("failed to create signature: %v", err)
	}

	response := &clients.Response{
		Choices: []clients.Choice{
			{
				Message: clients.Message{
					Role:    "assistant",
					Content: `{"answer": "Go is a programming language"}`,
				},
			},
		},
	}

	outputs, err := adapter.Parse(sig, response)
	if err != nil {
		t.Fatalf("failed to parse: %v", err)
	}

	if _, ok := outputs["answer"]; !ok {
		t.Error("expected 'answer' field in outputs")
	}

	if answer, ok := outputs["answer"].(string); ok {
		if answer != "Go is a programming language" {
			t.Errorf("expected specific answer, got: %s", answer)
		}
	} else {
		t.Error("answer should be a string")
	}
}

func TestBAMLAdapter_RenderTypeString(t *testing.T) {
	adapter := NewBAMLAdapter()

	tests := []struct {
		name     string
		field    *signatures.Field
		expected string
	}{
		{
			name:     "string type",
			field:    &signatures.Field{Type: "string"},
			expected: "string",
		},
		{
			name:     "int type",
			field:    &signatures.Field{Type: "int"},
			expected: "int",
		},
		{
			name:     "float type",
			field:    &signatures.Field{Type: "float"},
			expected: "float",
		},
		{
			name:     "boolean type",
			field:    &signatures.Field{Type: "bool"},
			expected: "boolean",
		},
		{
			name:     "string array",
			field:    &signatures.Field{Type: "[]string"},
			expected: "string[]",
		},
		{
			name:     "int array",
			field:    &signatures.Field{Type: "[]int"},
			expected: "int[]",
		},
		{
			name:     "map type",
			field:    &signatures.Field{Type: "map"},
			expected: "object",
		},
		{
			name:     "empty type defaults to string",
			field:    &signatures.Field{Type: ""},
			expected: "string",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.renderTypeString(tt.field)
			if result != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, result)
			}
		})
	}
}

func TestBAMLAdapter_FormatFieldValue(t *testing.T) {
	adapter := NewBAMLAdapter()

	tests := []struct {
		name     string
		value    interface{}
		contains string
	}{
		{
			name:     "string value",
			value:    "hello",
			contains: "hello",
		},
		{
			name:     "int value",
			value:    42,
			contains: "42",
		},
		{
			name:     "map value",
			value:    map[string]string{"key": "value"},
			contains: "key",
		},
		{
			name:     "slice value",
			value:    []string{"a", "b", "c"},
			contains: "a",
		},
		{
			name:     "nil value",
			value:    nil,
			contains: "null",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.formatFieldValue(tt.value)
			if !strings.Contains(result, tt.contains) {
				t.Errorf("expected result to contain %s, got: %s", tt.contains, result)
			}
		})
	}
}

func TestBAMLAdapter_BuildSystemMessage(t *testing.T) {
	adapter := NewBAMLAdapter()

	sig, err := signatures.NewSignature("question, context -> answer, reasoning")
	if err != nil {
		t.Fatalf("failed to create signature: %v", err)
	}
	sig.Instructions = "Be thorough in your reasoning."

	// Add descriptions
	sig.InputFields[0].Description = "The question to answer"
	sig.OutputFields[0].Description = "The answer to provide"

	systemMsg := adapter.buildBAMLSystemMessage(sig)

	// Check for key components
	expectedPhrases := []string{
		"Be thorough in your reasoning",
		"valid JSON",
		"Your input fields are",
		"Your output fields are",
		"question",
		"answer",
		"completed",
	}

	for _, phrase := range expectedPhrases {
		if !strings.Contains(systemMsg, phrase) {
			t.Errorf("system message should contain '%s'", phrase)
		}
	}
}

func TestBAMLAdapter_FieldStructure(t *testing.T) {
	adapter := NewBAMLAdapter()

	sig, err := signatures.NewSignature("input1, input2 -> output1, output2")
	if err != nil {
		t.Fatalf("failed to create signature: %v", err)
	}

	structure := adapter.formatFieldStructure(sig)

	// Check for BAML-style markers
	expectedMarkers := []string{
		"[[ ## input1 ## ]]",
		"[[ ## input2 ## ]]",
		"[[ ## output1 ## ]]",
		"[[ ## output2 ## ]]",
		"[[ ## completed ## ]]",
	}

	for _, marker := range expectedMarkers {
		if !strings.Contains(structure, marker) {
			t.Errorf("structure should contain marker '%s'", marker)
		}
	}

	// Check for type information
	if !strings.Contains(structure, "Output field") {
		t.Error("structure should contain output field type information")
	}
}

func TestBAMLAdapter_Name(t *testing.T) {
	adapter := NewBAMLAdapter()

	// BAMLAdapter extends JSONAdapter, so it should have json name
	if adapter.Name() != "json" {
		t.Errorf("expected adapter name 'json', got '%s'", adapter.Name())
	}
}

func TestBAMLAdapter_WithCustomCommentSymbol(t *testing.T) {
	adapter := NewBAMLAdapterWithCommentSymbol("//")

	if adapter.commentSymbol != "//" {
		t.Errorf("expected comment symbol '//', got '%s'", adapter.commentSymbol)
	}
}

func TestBAMLAdapter_ParseMalformedJSON(t *testing.T) {
	adapter := NewBAMLAdapter()

	sig, err := signatures.NewSignature("question -> answer")
	if err != nil {
		t.Fatalf("failed to create signature: %v", err)
	}

	// Test with malformed JSON wrapped in markdown
	response := &clients.Response{
		Choices: []clients.Choice{
			{
				Message: clients.Message{
					Role:    "assistant",
					Content: "```json\n{\"answer\": \"Go is great\"}\n```",
				},
			},
		},
	}

	outputs, err := adapter.Parse(sig, response)
	if err != nil {
		t.Fatalf("failed to parse: %v", err)
	}

	if _, ok := outputs["answer"]; !ok {
		t.Error("expected 'answer' field in outputs")
	}
}

func TestBAMLAdapter_InheritsJSONFeatures(t *testing.T) {
	adapter := NewBAMLAdapter()

	sig, err := signatures.NewSignature("question -> answer")
	if err != nil {
		t.Fatalf("failed to create signature: %v", err)
	}

	// Test that it can parse JSON (inherits from JSONAdapter)
	response := &clients.Response{
		Choices: []clients.Choice{
			{
				Message: clients.Message{
					Role:    "assistant",
					Content: `{"answer": "test", "extra": "field"}`,
				},
			},
		},
	}

	outputs, err := adapter.Parse(sig, response)
	if err != nil {
		t.Fatalf("failed to parse: %v", err)
	}

	// Should extract the answer field
	if _, ok := outputs["answer"]; !ok {
		t.Error("expected 'answer' field in outputs")
	}
}
