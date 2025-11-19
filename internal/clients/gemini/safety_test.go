package gemini

import (
	"testing"
)

func TestNewSafetySetting(t *testing.T) {
	setting := NewSafetySetting(SafetyCategoryHate, SafetyThresholdBlockMediumAndAbove)

	if setting.Category != string(SafetyCategoryHate) {
		t.Errorf("got category %q, want %q", setting.Category, SafetyCategoryHate)
	}

	if setting.Threshold != string(SafetyThresholdBlockMediumAndAbove) {
		t.Errorf("got threshold %q, want %q", setting.Threshold, SafetyThresholdBlockMediumAndAbove)
	}
}

func TestDefaultSafetySettings(t *testing.T) {
	settings := DefaultSafetySettings()

	if len(settings) != 4 {
		t.Errorf("got %d settings, want 4", len(settings))
	}

	// Verify all have BLOCK_MEDIUM_AND_ABOVE threshold
	for _, setting := range settings {
		if setting.Threshold != string(SafetyThresholdBlockMediumAndAbove) {
			t.Errorf("expected threshold %q, got %q", SafetyThresholdBlockMediumAndAbove, setting.Threshold)
		}
	}
}

func TestRelaxedSafetySettings(t *testing.T) {
	settings := RelaxedSafetySettings()

	if len(settings) != 4 {
		t.Errorf("got %d settings, want 4", len(settings))
	}

	// Verify all have BLOCK_ONLY_HIGH threshold
	for _, setting := range settings {
		if setting.Threshold != string(SafetyThresholdBlockOnlyHigh) {
			t.Errorf("expected threshold %q, got %q", SafetyThresholdBlockOnlyHigh, setting.Threshold)
		}
	}
}

func TestStrictSafetySettings(t *testing.T) {
	settings := StrictSafetySettings()

	if len(settings) != 4 {
		t.Errorf("got %d settings, want 4", len(settings))
	}

	// Verify all have BLOCK_LOW_AND_ABOVE threshold
	for _, setting := range settings {
		if setting.Threshold != string(SafetyThresholdBlockLowAndAbove) {
			t.Errorf("expected threshold %q, got %q", SafetyThresholdBlockLowAndAbove, setting.Threshold)
		}
	}
}

func TestNoSafetySettings(t *testing.T) {
	settings := NoSafetySettings()

	if len(settings) != 4 {
		t.Errorf("got %d settings, want 4", len(settings))
	}

	// Verify all have BLOCK_NONE threshold
	for _, setting := range settings {
		if setting.Threshold != string(SafetyThresholdBlockNone) {
			t.Errorf("expected threshold %q, got %q", SafetyThresholdBlockNone, setting.Threshold)
		}
	}
}

func TestGetAllSafetyCategories(t *testing.T) {
	categories := GetAllSafetyCategories()

	if len(categories) != 4 {
		t.Errorf("got %d categories, want 4", len(categories))
	}

	expectedCategories := []SafetyCategory{
		SafetyCategoryHate,
		SafetyCategorySexual,
		SafetyCategoryHarassment,
		SafetyCategoryDangerous,
	}

	for _, expected := range expectedCategories {
		found := false
		for _, cat := range categories {
			if cat == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected category %q not found", expected)
		}
	}
}

func TestGetAllSafetyThresholds(t *testing.T) {
	thresholds := GetAllSafetyThresholds()

	if len(thresholds) != 4 {
		t.Errorf("got %d thresholds, want 4", len(thresholds))
	}

	expectedThresholds := []SafetyThreshold{
		SafetyThresholdBlockLowAndAbove,
		SafetyThresholdBlockMediumAndAbove,
		SafetyThresholdBlockOnlyHigh,
		SafetyThresholdBlockNone,
	}

	for _, expected := range expectedThresholds {
		found := false
		for _, threshold := range thresholds {
			if threshold == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected threshold %q not found", expected)
		}
	}
}

func TestSafetyConstants(t *testing.T) {
	tests := []struct {
		name     string
		category SafetyCategory
		expected string
	}{
		{"hate speech", SafetyCategoryHate, "HARM_CATEGORY_HATE_SPEECH"},
		{"sexual", SafetyCategorySexual, "HARM_CATEGORY_SEXUALLY_EXPLICIT"},
		{"harassment", SafetyCategoryHarassment, "HARM_CATEGORY_HARASSMENT"},
		{"dangerous", SafetyCategoryDangerous, "HARM_CATEGORY_DANGEROUS_CONTENT"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if string(tt.category) != tt.expected {
				t.Errorf("got %q, want %q", tt.category, tt.expected)
			}
		})
	}
}
