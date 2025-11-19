package gemini

// SafetyCategory represents a category of safety concern.
type SafetyCategory string

const (
	// SafetyCategoryUnspecified is the default value.
	SafetyCategoryUnspecified SafetyCategory = "HARM_CATEGORY_UNSPECIFIED"

	// SafetyCategoryHate represents hate speech.
	SafetyCategoryHate SafetyCategory = "HARM_CATEGORY_HATE_SPEECH"

	// SafetyCategorySexual represents sexually explicit content.
	SafetyCategorySexual SafetyCategory = "HARM_CATEGORY_SEXUALLY_EXPLICIT"

	// SafetyCategoryHarassment represents harassment.
	SafetyCategoryHarassment SafetyCategory = "HARM_CATEGORY_HARASSMENT"

	// SafetyCategoryDangerous represents dangerous content.
	SafetyCategoryDangerous SafetyCategory = "HARM_CATEGORY_DANGEROUS_CONTENT"
)

// SafetyThreshold represents the threshold for blocking content.
type SafetyThreshold string

const (
	// SafetyThresholdUnspecified is the default value.
	SafetyThresholdUnspecified SafetyThreshold = "HARM_BLOCK_THRESHOLD_UNSPECIFIED"

	// SafetyThresholdBlockLowAndAbove blocks content with LOW and above probability.
	SafetyThresholdBlockLowAndAbove SafetyThreshold = "BLOCK_LOW_AND_ABOVE"

	// SafetyThresholdBlockMediumAndAbove blocks content with MEDIUM and above probability.
	SafetyThresholdBlockMediumAndAbove SafetyThreshold = "BLOCK_MEDIUM_AND_ABOVE"

	// SafetyThresholdBlockOnlyHigh blocks only content with HIGH probability.
	SafetyThresholdBlockOnlyHigh SafetyThreshold = "BLOCK_ONLY_HIGH"

	// SafetyThresholdBlockNone doesn't block any content.
	SafetyThresholdBlockNone SafetyThreshold = "BLOCK_NONE"
)

// NewSafetySetting creates a new safety setting.
func NewSafetySetting(category SafetyCategory, threshold SafetyThreshold) SafetySetting {
	return SafetySetting{
		Category:  string(category),
		Threshold: string(threshold),
	}
}

// DefaultSafetySettings returns the default safety settings.
func DefaultSafetySettings() []SafetySetting {
	return []SafetySetting{
		NewSafetySetting(SafetyCategoryHate, SafetyThresholdBlockMediumAndAbove),
		NewSafetySetting(SafetyCategorySexual, SafetyThresholdBlockMediumAndAbove),
		NewSafetySetting(SafetyCategoryHarassment, SafetyThresholdBlockMediumAndAbove),
		NewSafetySetting(SafetyCategoryDangerous, SafetyThresholdBlockMediumAndAbove),
	}
}

// RelaxedSafetySettings returns safety settings that are less restrictive.
func RelaxedSafetySettings() []SafetySetting {
	return []SafetySetting{
		NewSafetySetting(SafetyCategoryHate, SafetyThresholdBlockOnlyHigh),
		NewSafetySetting(SafetyCategorySexual, SafetyThresholdBlockOnlyHigh),
		NewSafetySetting(SafetyCategoryHarassment, SafetyThresholdBlockOnlyHigh),
		NewSafetySetting(SafetyCategoryDangerous, SafetyThresholdBlockOnlyHigh),
	}
}

// StrictSafetySettings returns safety settings that are more restrictive.
func StrictSafetySettings() []SafetySetting {
	return []SafetySetting{
		NewSafetySetting(SafetyCategoryHate, SafetyThresholdBlockLowAndAbove),
		NewSafetySetting(SafetyCategorySexual, SafetyThresholdBlockLowAndAbove),
		NewSafetySetting(SafetyCategoryHarassment, SafetyThresholdBlockLowAndAbove),
		NewSafetySetting(SafetyCategoryDangerous, SafetyThresholdBlockLowAndAbove),
	}
}

// NoSafetySettings returns safety settings that don't block any content.
func NoSafetySettings() []SafetySetting {
	return []SafetySetting{
		NewSafetySetting(SafetyCategoryHate, SafetyThresholdBlockNone),
		NewSafetySetting(SafetyCategorySexual, SafetyThresholdBlockNone),
		NewSafetySetting(SafetyCategoryHarassment, SafetyThresholdBlockNone),
		NewSafetySetting(SafetyCategoryDangerous, SafetyThresholdBlockNone),
	}
}

// GetAllSafetyCategories returns all available safety categories.
func GetAllSafetyCategories() []SafetyCategory {
	return []SafetyCategory{
		SafetyCategoryHate,
		SafetyCategorySexual,
		SafetyCategoryHarassment,
		SafetyCategoryDangerous,
	}
}

// GetAllSafetyThresholds returns all available safety thresholds.
func GetAllSafetyThresholds() []SafetyThreshold {
	return []SafetyThreshold{
		SafetyThresholdBlockLowAndAbove,
		SafetyThresholdBlockMediumAndAbove,
		SafetyThresholdBlockOnlyHigh,
		SafetyThresholdBlockNone,
	}
}
