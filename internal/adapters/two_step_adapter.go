package adapters

import (
	"context"
	"fmt"
	"strings"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

// TwoStepAdapter implements a two-phase adapter for improved structured output extraction.
// Phase 1: Uses a natural prompt format with the main LM for reasoning/generation.
// Phase 2: Uses an extraction model (typically smaller/cheaper) with ChatAdapter to extract structured data.
//
// This is particularly useful with reasoning models that may struggle with strict structured outputs.
type TwoStepAdapter struct {
	*BaseAdapter
	extractionModel *clients.LM
	chatAdapter     *ChatAdapter
}

// NewTwoStepAdapter creates a new two-step adapter with the specified extraction model.
// The extraction model is typically a smaller, cheaper model used for parsing structured outputs.
func NewTwoStepAdapter(extractionModel *clients.LM) *TwoStepAdapter {
	if extractionModel == nil {
		panic("extraction model cannot be nil")
	}
	return &TwoStepAdapter{
		BaseAdapter:     NewBaseAdapter("two_step"),
		extractionModel: extractionModel,
		chatAdapter:     NewChatAdapter(),
	}
}

// Format implements Adapter.Format for the first phase.
// Creates a natural language prompt that doesn't require strict structured output.
func (a *TwoStepAdapter) Format(sig *signatures.Signature, inputs map[string]interface{}, demos []map[string]interface{}) (*clients.Request, error) {
	request := clients.NewRequest()

	// Build task description for the main LM
	taskDesc := a.buildTaskDescription(sig)
	request.WithMessages(clients.NewMessage("system", taskDesc))

	// Add demonstrations in a natural format
	for _, demo := range demos {
		// Format demo input
		userMsg := a.formatUserMessage(sig, demo)
		request.WithMessages(clients.NewMessage("user", userMsg))

		// Format demo output in natural language
		assistantMsg := a.formatAssistantMessage(sig, demo)
		request.WithMessages(clients.NewMessage("assistant", assistantMsg))
	}

	// Add current input
	userMsg := a.formatUserMessage(sig, inputs)
	request.WithMessages(clients.NewMessage("user", userMsg))

	return request, nil
}

// Parse implements Adapter.Parse for the second phase.
// Uses the extraction model to parse structured data from the raw completion.
func (a *TwoStepAdapter) Parse(sig *signatures.Signature, response *clients.Response) (map[string]interface{}, error) {
	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	rawCompletion := response.Choices[0].Message.Content

	// Create an extractor signature: text -> {original output fields}
	extractorSig := a.createExtractorSignature(sig)

	// Format the extraction request
	extractionInputs := map[string]interface{}{
		"text": rawCompletion,
	}

	// Use ChatAdapter to format the extraction request
	extractionRequest, err := a.chatAdapter.Format(extractorSig, extractionInputs, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to format extraction request: %w", err)
	}

	// Call the extraction model
	ctx := context.Background()
	extractionResponse, err := a.extractionModel.Call(ctx, extractionRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to call extraction model: %w", err)
	}

	// Parse the extraction response using ChatAdapter
	outputs, err := a.chatAdapter.Parse(extractorSig, extractionResponse)
	if err != nil {
		return nil, fmt.Errorf("failed to parse extraction response: %w", err)
	}

	return outputs, nil
}

// buildTaskDescription creates a natural language task description.
func (a *TwoStepAdapter) buildTaskDescription(sig *signatures.Signature) string {
	var sb strings.Builder

	sb.WriteString("You are a helpful assistant that can solve tasks based on user input.\n\n")

	// Describe input fields
	if len(sig.InputFields) > 0 {
		sb.WriteString("As input, you will be provided with:\n")
		for _, field := range sig.InputFields {
			sb.WriteString(fmt.Sprintf("- %s", field.Name))
			if field.Description != "" {
				sb.WriteString(fmt.Sprintf(": %s", field.Description))
			}
			sb.WriteString("\n")
		}
		sb.WriteString("\n")
	}

	// Describe output requirements
	if len(sig.OutputFields) > 0 {
		sb.WriteString("Your outputs must contain:\n")
		for _, field := range sig.OutputFields {
			sb.WriteString(fmt.Sprintf("- %s", field.Name))
			if field.Description != "" {
				sb.WriteString(fmt.Sprintf(": %s", field.Description))
			}
			sb.WriteString("\n")
		}
		sb.WriteString("\n")
	}

	sb.WriteString("You should lay out your outputs in detail so that your answer can be understood by another agent.\n")

	// Add signature-specific instructions
	if sig.Instructions != "" {
		sb.WriteString(fmt.Sprintf("\nSpecific instructions: %s", sig.Instructions))
	}

	return sb.String()
}

// formatUserMessage formats input fields as a natural user message.
func (a *TwoStepAdapter) formatUserMessage(sig *signatures.Signature, data map[string]interface{}) string {
	var parts []string

	for _, field := range sig.InputFields {
		if val, ok := data[field.Name]; ok {
			parts = append(parts, fmt.Sprintf("%s: %v", field.Name, val))
		}
	}

	return strings.Join(parts, "\n\n")
}

// formatAssistantMessage formats output fields as a natural assistant message.
func (a *TwoStepAdapter) formatAssistantMessage(sig *signatures.Signature, data map[string]interface{}) string {
	var parts []string

	for _, field := range sig.OutputFields {
		if val, ok := data[field.Name]; ok {
			parts = append(parts, fmt.Sprintf("%s: %v", field.Name, val))
		}
	}

	return strings.Join(parts, "\n\n")
}

// createExtractorSignature creates a signature for extracting structured data from text.
// The signature has a single "text" input field and all the original output fields.
func (a *TwoStepAdapter) createExtractorSignature(originalSig *signatures.Signature) *signatures.Signature {
	// Create a text input field
	textField := signatures.NewInputField("text").
		WithDescription("The text to extract information from")

	// Build instructions for extraction
	outputNames := make([]string, len(originalSig.OutputFields))
	for i, field := range originalSig.OutputFields {
		outputNames[i] = fmt.Sprintf("`%s`", field.Name)
	}
	instructions := fmt.Sprintf(
		"The input is a text that should contain all the necessary information to produce the fields %s. "+
			"Your job is to extract the fields from the text verbatim. "+
			"Extract precisely the appropriate value (content) for each field.",
		strings.Join(outputNames, ", "),
	)

	// Create the extractor signature
	extractorSig := signatures.NewSignatureWithFields(
		[]*signatures.Field{textField},
		originalSig.OutputFields,
	).WithInstructions(instructions)

	return extractorSig
}
