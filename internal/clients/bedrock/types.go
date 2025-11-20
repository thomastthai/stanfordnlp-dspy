package bedrock

// BedrockError represents an error from AWS Bedrock.
type BedrockError struct {
	StatusCode int
	Code       string
	Message    string
	Retryable  bool
}

// Error implements the error interface.
func (e *BedrockError) Error() string {
	return e.Message
}

// ModelFamily represents a family of models on Bedrock.
type ModelFamily string

const (
	ModelFamilyAnthropic ModelFamily = "anthropic"
	ModelFamilyTitan     ModelFamily = "titan"
	ModelFamilyLlama     ModelFamily = "llama"
	ModelFamilyAI21      ModelFamily = "ai21"
	ModelFamilyCohere    ModelFamily = "cohere"
	ModelFamilyStability ModelFamily = "stability"
)

// ModelCapability represents capabilities of a Bedrock model.
type ModelCapability struct {
	TextGeneration  bool
	ChatCompletion  bool
	Embedding       bool
	ImageGeneration bool
	Streaming       bool
}

// GetModelCapabilities returns the capabilities for a model family.
func GetModelCapabilities(family ModelFamily) ModelCapability {
	switch family {
	case ModelFamilyAnthropic:
		return ModelCapability{
			TextGeneration:  true,
			ChatCompletion:  true,
			Embedding:       false,
			ImageGeneration: false,
			Streaming:       true,
		}
	case ModelFamilyTitan:
		return ModelCapability{
			TextGeneration:  true,
			ChatCompletion:  false,
			Embedding:       true,
			ImageGeneration: false,
			Streaming:       true,
		}
	case ModelFamilyLlama:
		return ModelCapability{
			TextGeneration:  true,
			ChatCompletion:  true,
			Embedding:       false,
			ImageGeneration: false,
			Streaming:       true,
		}
	case ModelFamilyAI21:
		return ModelCapability{
			TextGeneration:  true,
			ChatCompletion:  false,
			Embedding:       false,
			ImageGeneration: false,
			Streaming:       false,
		}
	case ModelFamilyCohere:
		return ModelCapability{
			TextGeneration:  true,
			ChatCompletion:  false,
			Embedding:       true,
			ImageGeneration: false,
			Streaming:       false,
		}
	default:
		return ModelCapability{}
	}
}

// ThrottlingConfig contains configuration for handling Bedrock throttling.
type ThrottlingConfig struct {
	MaxRetries        int
	InitialBackoff    int // milliseconds
	MaxBackoff        int // milliseconds
	BackoffMultiplier float64
}

// DefaultThrottlingConfig returns the default throttling configuration.
func DefaultThrottlingConfig() ThrottlingConfig {
	return ThrottlingConfig{
		MaxRetries:        5,
		InitialBackoff:    1000,  // 1 second
		MaxBackoff:        60000, // 60 seconds
		BackoffMultiplier: 2.0,
	}
}

// BedrockRegion represents an AWS region with Bedrock support.
type BedrockRegion string

const (
	RegionUSEast1      BedrockRegion = "us-east-1"
	RegionUSWest2      BedrockRegion = "us-west-2"
	RegionAPNortheast1 BedrockRegion = "ap-northeast-1"
	RegionAPSoutheast1 BedrockRegion = "ap-southeast-1"
	RegionAPSoutheast2 BedrockRegion = "ap-southeast-2"
	RegionEUCentral1   BedrockRegion = "eu-central-1"
	RegionEUWest1      BedrockRegion = "eu-west-1"
	RegionEUWest2      BedrockRegion = "eu-west-2"
	RegionEUWest3      BedrockRegion = "eu-west-3"
)

// ModelConfig contains model-specific configuration.
type ModelConfig struct {
	ModelID       string
	MaxTokens     int
	Temperature   float64
	TopP          float64
	TopK          int
	StopSequences []string
}

// ValidationError represents a validation error for Bedrock requests.
type ValidationError struct {
	Field   string
	Message string
}

// Error implements the error interface.
func (v *ValidationError) Error() string {
	return v.Message
}

// isRetryableAWSError checks if an AWS error is retryable.
func isRetryableAWSError(err error) bool {
	if err == nil {
		return false
	}

	errMsg := err.Error()

	// Check for common retryable errors
	retryableErrors := []string{
		"ThrottlingException",
		"ServiceUnavailable",
		"InternalServerException",
		"ModelNotReadyException",
		"TooManyRequestsException",
		"RequestTimeout",
		"ModelTimeoutException",
	}

	for _, retryable := range retryableErrors {
		if contains(errMsg, retryable) {
			return true
		}
	}

	return false
}

// contains checks if a string contains a substring.
func contains(s, substr string) bool {
	return len(s) >= len(substr) &&
		(s == substr || len(s) > len(substr) &&
			(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr ||
				indexContains(s, substr)))
}

// indexContains checks if string contains substring at any position.
func indexContains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
