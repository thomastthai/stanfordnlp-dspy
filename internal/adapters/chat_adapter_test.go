package adapters

import (
	"testing"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

func TestChatAdapter_Format(t *testing.T) {
	adapter := NewChatAdapter()

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

	if len(req.Messages) < 2 {
		t.Errorf("expected at least 2 messages (system + user), got %d", len(req.Messages))
	}

	// Check system message
	if req.Messages[0].Role != "system" {
		t.Errorf("expected first message to be system, got %s", req.Messages[0].Role)
	}

	// Check user message
	lastMsg := req.Messages[len(req.Messages)-1]
	if lastMsg.Role != "user" {
		t.Errorf("expected last message to be user, got %s", lastMsg.Role)
	}
}

func TestChatAdapter_Parse(t *testing.T) {
	adapter := NewChatAdapter()

	sig, err := signatures.NewSignature("question -> answer")
	if err != nil {
		t.Fatalf("failed to create signature: %v", err)
	}

	response := &clients.Response{
		Choices: []clients.Choice{
			{
				Message: clients.Message{
					Role:    "assistant",
					Content: "answer: Go is a programming language",
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

func TestChatAdapter_WithDemos(t *testing.T) {
	adapter := NewChatAdapter()

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

	// Should have: demo user + demo assistant + current user (no system message without instructions)
	if len(req.Messages) < 3 {
		t.Errorf("expected at least 3 messages with demos, got %d", len(req.Messages))
	}

	// Verify demo messages exist
	foundDemoUser := false
	foundDemoAssistant := false
	for i, msg := range req.Messages {
		if i < len(req.Messages)-1 { // Not the last message
			if msg.Role == "user" {
				foundDemoUser = true
			}
			if msg.Role == "assistant" {
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
