package databricks

// DatabricksError represents an error from Databricks API.
type DatabricksError struct {
	ErrorCode string `json:"error_code"`
	Message   string `json:"message"`
	Details   string `json:"details,omitempty"`
}

// Error implements the error interface.
func (e *DatabricksError) Error() string {
	if e.Details != "" {
		return e.Message + ": " + e.Details
	}
	return e.Message
}

// IsRetryable checks if the error is retryable.
func (e *DatabricksError) IsRetryable() bool {
	retryableCodes := map[string]bool{
		"RESOURCE_EXHAUSTED":     true,
		"TEMPORARILY_UNAVAILABLE": true,
		"INTERNAL_ERROR":          true,
		"REQUEST_LIMIT_EXCEEDED":  true,
		"DEADLINE_EXCEEDED":       true,
	}
	return retryableCodes[e.ErrorCode]
}

// FoundationModel represents a Databricks foundation model.
type FoundationModel string

const (
	// DBRX models
	ModelDBRXInstruct FoundationModel = "databricks-dbrx-instruct"
	
	// Meta Llama models
	ModelLlama3_70BInstruct FoundationModel = "databricks-meta-llama-3-70b-instruct"
	ModelLlama3_8BInstruct  FoundationModel = "databricks-meta-llama-3-8b-instruct"
	ModelLlama2_70BChat     FoundationModel = "databricks-meta-llama-2-70b-chat"
	
	// Mixtral models
	ModelMixtral8x7BInstruct FoundationModel = "databricks-mixtral-8x7b-instruct"
	
	// MPT models
	ModelMPT7BInstruct  FoundationModel = "databricks-mpt-7b-instruct"
	ModelMPT30BInstruct FoundationModel = "databricks-mpt-30b-instruct"
)

// ModelCapability represents capabilities of a model.
type ModelCapability struct {
	ChatCompletion bool
	Completion     bool
	Embedding      bool
	Streaming      bool
}

// GetModelCapabilities returns the capabilities for a foundation model.
func GetModelCapabilities(model FoundationModel) ModelCapability {
	switch model {
	case ModelDBRXInstruct, ModelLlama3_70BInstruct, ModelLlama3_8BInstruct,
		ModelLlama2_70BChat, ModelMixtral8x7BInstruct,
		ModelMPT7BInstruct, ModelMPT30BInstruct:
		return ModelCapability{
			ChatCompletion: true,
			Completion:     true,
			Embedding:      false,
			Streaming:      true,
		}
	default:
		return ModelCapability{
			ChatCompletion: false,
			Completion:     true,
			Embedding:      false,
			Streaming:      false,
		}
	}
}

// EndpointType represents the type of serving endpoint.
type EndpointType string

const (
	EndpointTypeFoundationModel EndpointType = "foundation_model"
	EndpointTypeCustomModel     EndpointType = "custom_model"
	EndpointTypeExternalModel   EndpointType = "external_model"
)

// WorkloadSize represents the compute size for a serving endpoint.
type WorkloadSize string

const (
	WorkloadSizeSmall  WorkloadSize = "Small"
	WorkloadSizeMedium WorkloadSize = "Medium"
	WorkloadSizeLarge  WorkloadSize = "Large"
)

// RateLimitConfig contains rate limit configuration.
type RateLimitConfig struct {
	Calls           int `json:"calls"`
	RenewalPeriod   int `json:"renewal_period"`
	Key             string `json:"key,omitempty"`
}

// Workspace represents a Databricks workspace.
type Workspace struct {
	ID        string `json:"workspace_id"`
	Name      string `json:"workspace_name"`
	Deployment string `json:"deployment_name"`
	Region    string `json:"aws_region,omitempty"`
}

// TokenInfo contains information about a Databricks token.
type TokenInfo struct {
	TokenID      string `json:"token_id"`
	CreationTime int64  `json:"creation_time"`
	ExpiryTime   int64  `json:"expiry_time"`
	Comment      string `json:"comment,omitempty"`
}

// ServingEndpointPermission represents permissions for a serving endpoint.
type ServingEndpointPermission struct {
	UserName          string `json:"user_name,omitempty"`
	GroupName         string `json:"group_name,omitempty"`
	ServicePrincipal  string `json:"service_principal_name,omitempty"`
	PermissionLevel   string `json:"permission_level"`
}

// PermissionLevel represents levels of access to serving endpoints.
type PermissionLevel string

const (
	PermissionLevelCanManage PermissionLevel = "CAN_MANAGE"
	PermissionLevelCanQuery  PermissionLevel = "CAN_QUERY"
	PermissionLevelCanView   PermissionLevel = "CAN_VIEW"
)

// ContentFilterResult represents content filtering results.
type ContentFilterResult struct {
	Filtered bool   `json:"filtered"`
	Reason   string `json:"reason,omitempty"`
}
