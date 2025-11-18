package adapters

import (
	"fmt"
	"strings"

	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

// ChatAdapter formats requests for chat-based LMs.
type ChatAdapter struct {
	*BaseAdapter
}

// NewChatAdapter creates a new chat adapter.
func NewChatAdapter() *ChatAdapter {
	return &ChatAdapter{
		BaseAdapter: NewBaseAdapter("chat"),
	}
}

// Format implements Adapter.Format.
func (a *ChatAdapter) Format(sig *signatures.Signature, inputs map[string]interface{}, demos []map[string]interface{}) (*clients.Request, error) {
	request := clients.NewRequest()

	// Add system message with instructions
	if sig.Instructions != "" {
		request.WithMessages(clients.NewMessage("system", sig.Instructions))
	}

	// Add demonstrations as few-shot examples
	for _, demo := range demos {
		// Format demo input
		inputParts := []string{}
		for _, field := range sig.InputFields {
			if val, ok := demo[field.Name]; ok {
				inputParts = append(inputParts, fmt.Sprintf("%s %v", field.Prefix, val))
			}
		}
		if len(inputParts) > 0 {
			request.WithMessages(clients.NewMessage("user", strings.Join(inputParts, "\n")))
		}

		// Format demo output
		outputParts := []string{}
		for _, field := range sig.OutputFields {
			if val, ok := demo[field.Name]; ok {
				outputParts = append(outputParts, fmt.Sprintf("%s %v", field.Prefix, val))
			}
		}
		if len(outputParts) > 0 {
			request.WithMessages(clients.NewMessage("assistant", strings.Join(outputParts, "\n")))
		}
	}

	// Add user message with current inputs
	inputParts := []string{}
	for _, field := range sig.InputFields {
		if val, ok := inputs[field.Name]; ok {
			inputParts = append(inputParts, fmt.Sprintf("%s %v", field.Prefix, val))
		}
	}

	if len(inputParts) == 0 {
		return nil, fmt.Errorf("no input fields provided")
	}

	request.WithMessages(clients.NewMessage("user", strings.Join(inputParts, "\n")))

	return request, nil
}

// Parse implements Adapter.Parse.
func (a *ChatAdapter) Parse(sig *signatures.Signature, response *clients.Response) (map[string]interface{}, error) {
	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	content := response.Choices[0].Message.Content
	outputs := make(map[string]interface{})

	// Simple parsing: look for field prefixes in the response
	for _, field := range sig.OutputFields {
		// Try to extract the field value
		// For now, use a simple heuristic
		if strings.Contains(content, field.Prefix) {
			parts := strings.Split(content, field.Prefix)
			if len(parts) > 1 {
				// Take the text after the prefix until the next field or end
				value := strings.TrimSpace(parts[1])

				// Find the next field prefix or end
				for _, nextField := range sig.OutputFields {
					if nextField.Name != field.Name && strings.Contains(value, nextField.Prefix) {
						parts := strings.Split(value, nextField.Prefix)
						value = strings.TrimSpace(parts[0])
						break
					}
				}

				outputs[field.Name] = value
			}
		}
	}

	// If no structured output found, use the whole response for the first output field
	if len(outputs) == 0 && len(sig.OutputFields) > 0 {
		outputs[sig.OutputFields[0].Name] = strings.TrimSpace(content)
	}

	return outputs, nil
}
