package clients

import (
	"fmt"
	"sync"
)

// Provider is a factory for creating language model clients.
// This follows the Terraform provider pattern.
type Provider interface {
	// Create creates a new language model client with the given configuration.
	Create(config map[string]interface{}) (BaseLM, error)

	// Name returns the provider name.
	Name() string

	// SupportedModels returns the list of supported model names.
	SupportedModels() []string
}

var (
	// Global registry of providers
	providerRegistry = make(map[string]Provider)
	providerMux      sync.RWMutex
)

// RegisterProvider registers a new provider.
func RegisterProvider(name string, provider Provider) {
	providerMux.Lock()
	defer providerMux.Unlock()
	providerRegistry[name] = provider
}

// GetProvider retrieves a provider by name.
func GetProvider(name string) (Provider, error) {
	providerMux.RLock()
	defer providerMux.RUnlock()

	provider, ok := providerRegistry[name]
	if !ok {
		return nil, fmt.Errorf("provider not found: %s", name)
	}

	return provider, nil
}

// ListProviders returns all registered provider names.
func ListProviders() []string {
	providerMux.RLock()
	defer providerMux.RUnlock()

	names := make([]string, 0, len(providerRegistry))
	for name := range providerRegistry {
		names = append(names, name)
	}

	return names
}

// CreateClient creates a language model client using the specified provider.
func CreateClient(providerName string, config map[string]interface{}) (BaseLM, error) {
	provider, err := GetProvider(providerName)
	if err != nil {
		return nil, err
	}

	return provider.Create(config)
}
