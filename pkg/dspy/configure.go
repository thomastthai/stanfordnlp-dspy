package dspy

// Configure updates the global DSPy settings.
// This is the primary way to set up DSPy for use.
//
// Example:
//
//	dspy.Configure(
//	    dspy.WithLM("openai/gpt-4"),
//	    dspy.WithTemperature(0.7),
//	    dspy.WithMaxTokens(500),
//	)
func Configure(opts ...SettingsOption) {
	settings := GetSettings()
	for _, opt := range opts {
		opt(settings)
	}
}
