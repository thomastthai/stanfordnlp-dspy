package dspy

import (
	"context"
	"testing"
	"time"
)

func TestNewSettings(t *testing.T) {
	s := NewSettings()
	
	if s.Temperature != 0.0 {
		t.Errorf("expected default temperature 0.0, got %f", s.Temperature)
	}
	
	if s.MaxTokens != 1000 {
		t.Errorf("expected default max tokens 1000, got %d", s.MaxTokens)
	}
	
	if !s.EnableCache {
		t.Error("expected cache to be enabled by default")
	}
	
	if s.Timeout != 30*time.Second {
		t.Errorf("expected default timeout 30s, got %v", s.Timeout)
	}
}

func TestSettingsCopy(t *testing.T) {
	s := NewSettings()
	s.Temperature = 0.7
	s.MaxTokens = 500
	s.Experimental["test"] = "value"
	
	s2 := s.Copy()
	
	if s2.Temperature != s.Temperature {
		t.Errorf("temperature not copied: got %f, want %f", s2.Temperature, s.Temperature)
	}
	
	if s2.MaxTokens != s.MaxTokens {
		t.Errorf("max tokens not copied: got %d, want %d", s2.MaxTokens, s.MaxTokens)
	}
	
	// Verify deep copy of experimental map
	if s2.Experimental["test"] != "value" {
		t.Error("experimental map not copied")
	}
	
	// Modify copy, should not affect original
	s2.Temperature = 0.9
	if s.Temperature == 0.9 {
		t.Error("modifying copy affected original")
	}
}

func TestSettingsOptions(t *testing.T) {
	tests := []struct {
		name   string
		option SettingsOption
		check  func(*Settings) error
	}{
		{
			name:   "WithTemperature",
			option: WithTemperature(0.8),
			check: func(s *Settings) error {
				if s.Temperature != 0.8 {
					t.Errorf("expected temperature 0.8, got %f", s.Temperature)
				}
				return nil
			},
		},
		{
			name:   "WithMaxTokens",
			option: WithMaxTokens(2000),
			check: func(s *Settings) error {
				if s.MaxTokens != 2000 {
					t.Errorf("expected max tokens 2000, got %d", s.MaxTokens)
				}
				return nil
			},
		},
		{
			name:   "WithCache",
			option: WithCache(false),
			check: func(s *Settings) error {
				if s.EnableCache {
					t.Error("expected cache to be disabled")
				}
				return nil
			},
		},
		{
			name:   "WithTimeout",
			option: WithTimeout(60 * time.Second),
			check: func(s *Settings) error {
				if s.Timeout != 60*time.Second {
					t.Errorf("expected timeout 60s, got %v", s.Timeout)
				}
				return nil
			},
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewSettings()
			tt.option(s)
			tt.check(s)
		})
	}
}

func TestConfigure(t *testing.T) {
	// Save original settings
	original := GetSettings().Copy()
	defer SetSettings(original)
	
	Configure(
		WithTemperature(0.5),
		WithMaxTokens(1500),
	)
	
	s := GetSettings()
	if s.Temperature != 0.5 {
		t.Errorf("expected temperature 0.5, got %f", s.Temperature)
	}
	
	if s.MaxTokens != 1500 {
		t.Errorf("expected max tokens 1500, got %d", s.MaxTokens)
	}
}

func TestContext(t *testing.T) {
	// Create context with custom settings
	ctx := Context(
		context.Background(),
		WithTemperature(0.9),
		WithMaxTokens(500),
	)
	
	// Extract settings from context
	s := SettingsFromContext(ctx)
	
	if s.Temperature != 0.9 {
		t.Errorf("expected temperature 0.9, got %f", s.Temperature)
	}
	
	if s.MaxTokens != 500 {
		t.Errorf("expected max tokens 500, got %d", s.MaxTokens)
	}
	
	// Verify global settings unchanged
	global := GetSettings()
	if global.Temperature == 0.9 {
		t.Error("global settings should not be affected by context")
	}
}

func TestSettingsFromContext_NoSettings(t *testing.T) {
	ctx := context.Background()
	s := SettingsFromContext(ctx)
	
	// Should return global settings
	if s == nil {
		t.Fatal("expected non-nil settings")
	}
}
