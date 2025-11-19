package azure

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// AuthProvider defines the interface for Azure authentication.
type AuthProvider interface {
	// GetToken returns a valid access token for Azure OpenAI.
	GetToken(ctx context.Context) (string, error)
}

// APIKeyAuth provides API key-based authentication.
type APIKeyAuth struct {
	apiKey string
}

// NewAPIKeyAuth creates a new API key authentication provider.
func NewAPIKeyAuth(apiKey string) *APIKeyAuth {
	return &APIKeyAuth{
		apiKey: apiKey,
	}
}

// GetToken returns the API key (which is used directly as a header).
func (a *APIKeyAuth) GetToken(ctx context.Context) (string, error) {
	if a.apiKey == "" {
		return "", fmt.Errorf("API key is empty")
	}
	return a.apiKey, nil
}

// TokenCredential represents an Azure AD token credential.
type TokenCredential struct {
	Token     string
	ExpiresAt time.Time
}

// IsValid checks if the token is still valid.
func (tc *TokenCredential) IsValid() bool {
	// Consider token expired if it expires in less than 5 minutes
	return time.Now().Add(5 * time.Minute).Before(tc.ExpiresAt)
}

// AzureADAuth provides Azure AD token-based authentication.
type AzureADAuth struct {
	tokenProvider func(ctx context.Context) (*TokenCredential, error)
	token         *TokenCredential
	mu            sync.RWMutex
}

// NewAzureADAuth creates a new Azure AD authentication provider.
// The tokenProvider function should handle obtaining tokens from Azure AD.
func NewAzureADAuth(tokenProvider func(ctx context.Context) (*TokenCredential, error)) *AzureADAuth {
	return &AzureADAuth{
		tokenProvider: tokenProvider,
	}
}

// GetToken returns a valid Azure AD access token, refreshing if necessary.
func (a *AzureADAuth) GetToken(ctx context.Context) (string, error) {
	// Check if we have a valid cached token
	a.mu.RLock()
	if a.token != nil && a.token.IsValid() {
		token := a.token.Token
		a.mu.RUnlock()
		return token, nil
	}
	a.mu.RUnlock()

	// Need to get a new token
	a.mu.Lock()
	defer a.mu.Unlock()

	// Double-check in case another goroutine already refreshed
	if a.token != nil && a.token.IsValid() {
		return a.token.Token, nil
	}

	// Get new token
	newToken, err := a.tokenProvider(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to obtain Azure AD token: %w", err)
	}

	a.token = newToken
	return newToken.Token, nil
}

// ManagedIdentityAuth provides authentication using Azure Managed Identity.
// This is a placeholder for actual Managed Identity implementation.
type ManagedIdentityAuth struct {
	clientID string
}

// NewManagedIdentityAuth creates a new Managed Identity authentication provider.
func NewManagedIdentityAuth(clientID string) *ManagedIdentityAuth {
	return &ManagedIdentityAuth{
		clientID: clientID,
	}
}

// GetToken obtains a token from Azure Managed Identity.
func (m *ManagedIdentityAuth) GetToken(ctx context.Context) (string, error) {
	// This is a placeholder - actual implementation would use Azure SDK
	// to obtain tokens from the Azure Instance Metadata Service (IMDS)
	return "", fmt.Errorf("Managed Identity authentication not yet implemented - use Azure SDK")
}

// ClientWithAuth extends the Client to support pluggable authentication.
type ClientWithAuth struct {
	*Client
	authProvider AuthProvider
}

// NewClientWithAuth creates a new Azure OpenAI client with custom authentication.
func NewClientWithAuth(opts ClientOptions, authProvider AuthProvider) (*ClientWithAuth, error) {
	// Create base client without validating API key if using custom auth
	if authProvider != nil && opts.APIKey == "" {
		opts.APIKey = "placeholder" // Will be replaced by auth provider
	}

	client, err := NewClient(opts)
	if err != nil {
		return nil, err
	}

	return &ClientWithAuth{
		Client:       client,
		authProvider: authProvider,
	}, nil
}

// getAuthHeader returns the appropriate authentication header value.
func (c *ClientWithAuth) getAuthHeader(ctx context.Context) (string, error) {
	if c.authProvider != nil {
		return c.authProvider.GetToken(ctx)
	}
	// Fall back to API key from client
	return c.apiKey, nil
}

// Note: The actual Call methods would need to be updated to use getAuthHeader()
// instead of directly using c.apiKey. This would require modifications to
// the client.go file to support the auth provider pattern.
