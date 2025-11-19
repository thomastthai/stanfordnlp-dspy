// Package predict provides prediction modules that interact with language models.
package predict

import (
	"context"
	"fmt"

	"github.com/stanfordnlp/dspy/internal/adapters"
	"github.com/stanfordnlp/dspy/internal/clients"
	"github.com/stanfordnlp/dspy/internal/signatures"
	"github.com/stanfordnlp/dspy/pkg/dspy"
)

// LMIntegration provides helper methods for integrating with language models.
// It encapsulates LM access, adapter usage, and request/response handling.
type LMIntegration struct {
	lm      clients.BaseLM
	adapter adapters.Adapter
}

// NewLMIntegration creates a new LM integration helper from context.
// It extracts the LM and adapter from the DSPy settings.
func NewLMIntegration(ctx context.Context) (*LMIntegration, error) {
	settings := dspy.SettingsFromContext(ctx)
	
	// Get LM from settings
	if settings.LM == nil {
		return nil, fmt.Errorf("no language model configured in settings")
	}
	
	lm, ok := settings.LM.(clients.BaseLM)
	if !ok {
		return nil, fmt.Errorf("configured LM does not implement BaseLM interface")
	}
	
	// Get or create adapter
	adapter := getAdapter(settings)
	
	return &LMIntegration{
		lm:      lm,
		adapter: adapter,
	}, nil
}

// Generate calls the LM with the given signature and inputs.
// It handles formatting the request via the adapter and parsing the response.
func (lmi *LMIntegration) Generate(ctx context.Context, sig *signatures.Signature,
	inputs map[string]interface{}, demos []map[string]interface{}) (map[string]interface{}, error) {
	
	// Format request using adapter
	request, err := lmi.adapter.Format(sig, inputs, demos)
	if err != nil {
		return nil, fmt.Errorf("failed to format request: %w", err)
	}
	
	// Apply settings to request
	settings := dspy.SettingsFromContext(ctx)
	if request.Temperature == 0 {
		request.Temperature = settings.Temperature
	}
	if request.MaxTokens == 0 || request.MaxTokens == 1000 {
		request.MaxTokens = settings.MaxTokens
	}
	
	// Call LM
	response, err := lmi.lm.Call(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("LM call failed: %w", err)
	}
	
	// Parse response using adapter
	output, err := lmi.adapter.Parse(sig, response)
	if err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	
	return output, nil
}

// Call provides direct access to the underlying LM for advanced use cases.
func (lmi *LMIntegration) Call(ctx context.Context, request *clients.Request) (*clients.Response, error) {
	return lmi.lm.Call(ctx, request)
}

// getAdapter returns the appropriate adapter based on settings.
func getAdapter(settings *dspy.Settings) adapters.Adapter {
	// If adapter is specified in settings, try to use it
	if settings.Adapter != "" {
		switch settings.Adapter {
		case "chat":
			return adapters.NewChatAdapter()
		case "json":
			return adapters.NewJSONAdapter()
		case "xml":
			return adapters.NewXMLAdapter()
		}
	}
	
	// Default to chat adapter
	return adapters.NewChatAdapter()
}
