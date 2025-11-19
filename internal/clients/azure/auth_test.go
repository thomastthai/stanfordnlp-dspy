package azure

import (
	"context"
	"testing"
	"time"
)

func TestAPIKeyAuth_GetToken(t *testing.T) {
	apiKey := "test-api-key"
	auth := NewAPIKeyAuth(apiKey)

	token, err := auth.GetToken(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if token != apiKey {
		t.Errorf("expected token '%s', got '%s'", apiKey, token)
	}
}

func TestAPIKeyAuth_GetToken_Empty(t *testing.T) {
	auth := NewAPIKeyAuth("")

	_, err := auth.GetToken(context.Background())
	if err == nil {
		t.Error("expected error for empty API key")
	}
}

func TestTokenCredential_IsValid(t *testing.T) {
	tests := []struct {
		name      string
		expiresAt time.Time
		want      bool
	}{
		{
			name:      "valid token",
			expiresAt: time.Now().Add(10 * time.Minute),
			want:      true,
		},
		{
			name:      "expired token",
			expiresAt: time.Now().Add(-10 * time.Minute),
			want:      false,
		},
		{
			name:      "token expiring soon",
			expiresAt: time.Now().Add(2 * time.Minute),
			want:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cred := &TokenCredential{
				Token:     "test-token",
				ExpiresAt: tt.expiresAt,
			}

			if got := cred.IsValid(); got != tt.want {
				t.Errorf("IsValid() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAzureADAuth_GetToken(t *testing.T) {
	called := false
	tokenProvider := func(ctx context.Context) (*TokenCredential, error) {
		called = true
		return &TokenCredential{
			Token:     "test-token",
			ExpiresAt: time.Now().Add(1 * time.Hour),
		}, nil
	}

	auth := NewAzureADAuth(tokenProvider)

	token, err := auth.GetToken(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if token != "test-token" {
		t.Errorf("expected token 'test-token', got '%s'", token)
	}

	if !called {
		t.Error("token provider was not called")
	}
}

func TestAzureADAuth_GetToken_Cached(t *testing.T) {
	callCount := 0
	tokenProvider := func(ctx context.Context) (*TokenCredential, error) {
		callCount++
		return &TokenCredential{
			Token:     "test-token",
			ExpiresAt: time.Now().Add(1 * time.Hour),
		}, nil
	}

	auth := NewAzureADAuth(tokenProvider)

	// First call should invoke provider
	_, err := auth.GetToken(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if callCount != 1 {
		t.Errorf("expected 1 call to token provider, got %d", callCount)
	}

	// Second call should use cached token
	_, err = auth.GetToken(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if callCount != 1 {
		t.Errorf("expected 1 call to token provider (cached), got %d", callCount)
	}
}

func TestAzureADAuth_GetToken_Refresh(t *testing.T) {
	callCount := 0
	tokenProvider := func(ctx context.Context) (*TokenCredential, error) {
		callCount++
		// Return token that expires soon to force refresh
		return &TokenCredential{
			Token:     "test-token",
			ExpiresAt: time.Now().Add(3 * time.Minute),
		}, nil
	}

	auth := NewAzureADAuth(tokenProvider)

	// First call
	_, err := auth.GetToken(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Second call should refresh (token expires in < 5 minutes)
	_, err = auth.GetToken(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if callCount != 2 {
		t.Errorf("expected 2 calls to token provider (refresh), got %d", callCount)
	}
}
