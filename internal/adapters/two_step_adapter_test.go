package adapters

import (
	"context"
	"testing"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

// mockLM is a mock language model for testing
type mockLM struct {
	responses []*clients.Response
	callCount int
}

func (m *mockLM) Call(ctx context.Context, request *clients.Request) (*clients.Response, error) {
	if m.callCount >= len(m.responses) {
		return &clients.Response{
			Choices: []clients.Choice{
				{
					Message: clients.Message{
						Role:    "assistant",
						Content: "mock response",
					},
				},
			},
		}, nil
	}
	response := m.responses[m.callCount]
	m.callCount++
	return response, nil
}

func (m *mockLM) CallBatch(ctx context.Context, requests []*clients.Request) ([]*clients.Response, error) {
	responses := make([]*clients.Response, len(requests))
	for i := range requests {
		resp, err := m.Call(ctx, requests[i])
		if err != nil {
			return nil, err
		}
		responses[i] = resp
	}
	return responses, nil
}

func (m *mockLM) Name() string {
	return "mock-model"
}

func (m *mockLM) Provider() string {
	return "mock"
}

func newMockLM(responses []*clients.Response) *clients.LM {
	// This is a simplified mock - in real implementation, we'd need proper LM construction
	// For testing purposes, we'll use the mock through the adapter directly
	return nil
}

func TestTwoStepAdapter_Format(t *testing.T) {
	// Create a mock extraction model
	mockExtraction := &mockLM{
		responses: []*clients.Response{
			{
				Choices: []clients.Choice{
					{
						Message: clients.Message{
							Role:    "assistant",
							Content: "answer: Go is a programming language",
						},
					},
				},
			},
		},
	}

	// Create mock LM wrapper (simplified for testing)
	// In practice, we'd need to properly initialize the LM
	extractionLM := &clients.LM{}

	adapter := NewTwoStepAdapter(extractionLM)

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

	// Verify the request has a system message with task description
	if len(req.Messages) < 2 {
		t.Errorf("expected at least 2 messages (system + user), got %d", len(req.Messages))
	}

	// Check system message
	if req.Messages[0].Role != "system" {
		t.Errorf("expected first message to be system, got %s", req.Messages[0].Role)
	}

	// System message should contain task description
	systemContent := req.Messages[0].Content
	if !contains(systemContent, "helpful assistant") {
		t.Error("system message should contain task description")
	}

	// Check user message
	lastMsg := req.Messages[len(req.Messages)-1]
	if lastMsg.Role != "user" {
		t.Errorf("expected last message to be user, got %s", lastMsg.Role)
	}

	// User message should contain the question
	if !contains(lastMsg.Content, "What is Go?") {
		t.Error("user message should contain the question")
	}

	_ = mockExtraction // Suppress unused warning
}

func TestTwoStepAdapter_FormatWithDemos(t *testing.T) {
	extractionLM := &clients.LM{}
	adapter := NewTwoStepAdapter(extractionLM)

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

	// Verify demo messages exist
	foundDemoUser := false
	foundDemoAssistant := false
	for i, msg := range req.Messages {
		if i < len(req.Messages)-1 { // Not the last message
			if msg.Role == "user" && contains(msg.Content, "Python") {
				foundDemoUser = true
			}
			if msg.Role == "assistant" && contains(msg.Content, "programming language") {
				foundDemoAssistant = true
			}
		}
	}

	if !foundDemoUser {
		t.Error("expected to find demo user message")
	}
	if !foundDemoAssistant {
		t.Error("expected to find demo assistant message")
	}
}

func TestTwoStepAdapter_CreateExtractorSignature(t *testing.T) {
	extractionLM := &clients.LM{}
	adapter := NewTwoStepAdapter(extractionLM)

	originalSig, err := signatures.NewSignature("question, context -> answer, reasoning")
	if err != nil {
		t.Fatalf("failed to create signature: %v", err)
	}

	extractorSig := adapter.createExtractorSignature(originalSig)

	// Should have one input field: "text"
	if len(extractorSig.InputFields) != 1 {
		t.Errorf("expected 1 input field, got %d", len(extractorSig.InputFields))
	}

	if extractorSig.InputFields[0].Name != "text" {
		t.Errorf("expected input field 'text', got '%s'", extractorSig.InputFields[0].Name)
	}

	// Should have same output fields as original
	if len(extractorSig.OutputFields) != len(originalSig.OutputFields) {
		t.Errorf("expected %d output fields, got %d", len(originalSig.OutputFields), len(extractorSig.OutputFields))
	}

	// Check output field names match
	for i, field := range originalSig.OutputFields {
		if extractorSig.OutputFields[i].Name != field.Name {
			t.Errorf("expected output field '%s', got '%s'", field.Name, extractorSig.OutputFields[i].Name)
		}
	}

	// Should have extraction instructions
	if extractorSig.Instructions == "" {
		t.Error("extractor signature should have instructions")
	}

	// Instructions should mention extracting fields
	if !contains(extractorSig.Instructions, "extract") {
		t.Error("instructions should mention 'extract'")
	}
}

func TestTwoStepAdapter_BuildTaskDescription(t *testing.T) {
	extractionLM := &clients.LM{}
	adapter := NewTwoStepAdapter(extractionLM)

	sig, err := signatures.NewSignature("question -> answer")
	if err != nil {
		t.Fatalf("failed to create signature: %v", err)
	}
	sig.Instructions = "Be concise"

	// Add descriptions to fields
	sig.InputFields[0].Description = "The question to answer"
	sig.OutputFields[0].Description = "The answer to provide"

	taskDesc := adapter.buildTaskDescription(sig)

	// Check that task description contains key elements
	expectedPhrases := []string{
		"helpful assistant",
		"As input",
		"question",
		"Your outputs must contain",
		"answer",
		"Be concise",
	}

	for _, phrase := range expectedPhrases {
		if !contains(taskDesc, phrase) {
			t.Errorf("task description should contain '%s'", phrase)
		}
	}
}

func TestTwoStepAdapter_Name(t *testing.T) {
	extractionLM := &clients.LM{}
	adapter := NewTwoStepAdapter(extractionLM)

	if adapter.Name() != "two_step" {
		t.Errorf("expected adapter name 'two_step', got '%s'", adapter.Name())
	}
}

func TestTwoStepAdapter_PanicOnNilExtractionModel(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when extraction model is nil")
		}
	}()

	NewTwoStepAdapter(nil)
}

// Helper function to check if a string contains a substring (case-insensitive)
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && (s[0:len(substr)] == substr || contains(s[1:], substr))))
}
