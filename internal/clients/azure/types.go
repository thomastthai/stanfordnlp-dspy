package azure

// AzureErrorResponse represents an error response from Azure OpenAI.
type AzureErrorResponse struct {
	Error AzureError `json:"error"`
}

// AzureError contains error details from Azure OpenAI.
type AzureError struct {
	Code       string      `json:"code"`
	Message    string      `json:"message"`
	Type       string      `json:"type"`
	Param      string      `json:"param,omitempty"`
	InnerError *InnerError `json:"innererror,omitempty"`
}

// InnerError contains additional error information.
type InnerError struct {
	Code                 string `json:"code,omitempty"`
	ContentFilterResults string `json:"content_filter_results,omitempty"`
}

// DeploymentInfo contains information about an Azure OpenAI deployment.
type DeploymentInfo struct {
	ID         string           `json:"id"`
	Model      string           `json:"model"`
	Status     string           `json:"status"`
	ScaleType  string           `json:"scale_type"`
	Capacity   int              `json:"capacity,omitempty"`
	Properties *DeploymentProps `json:"properties,omitempty"`
}

// DeploymentProps contains deployment properties.
type DeploymentProps struct {
	ModelID  string `json:"model_id"`
	Format   string `json:"model_format"`
	Version  string `json:"model_version"`
	Endpoint string `json:"endpoint,omitempty"`
}

// RateLimitInfo contains rate limit information from Azure OpenAI.
type RateLimitInfo struct {
	Requests          int `json:"requests"`
	RequestsRemaining int `json:"requests_remaining"`
	Tokens            int `json:"tokens"`
	TokensRemaining   int `json:"tokens_remaining"`
}

// ContentFilterResult represents content filtering results.
type ContentFilterResult struct {
	Hate     *FilterScore `json:"hate,omitempty"`
	SelfHarm *FilterScore `json:"self_harm,omitempty"`
	Sexual   *FilterScore `json:"sexual,omitempty"`
	Violence *FilterScore `json:"violence,omitempty"`
}

// FilterScore represents a content filter score.
type FilterScore struct {
	Filtered bool   `json:"filtered"`
	Severity string `json:"severity"` // "safe", "low", "medium", "high"
}

// APIVersion contains supported Azure OpenAI API versions.
type APIVersion string

const (
	// APIVersion2024_02_15 is the 2024-02-15-preview API version
	APIVersion2024_02_15 APIVersion = "2024-02-15-preview"
	
	// APIVersion2024_06_01 is the 2024-06-01 API version
	APIVersion2024_06_01 APIVersion = "2024-06-01"
	
	// APIVersion2024_10_21 is the 2024-10-21 API version
	APIVersion2024_10_21 APIVersion = "2024-10-21"
)

// DeploymentCapability represents capabilities of an Azure deployment.
type DeploymentCapability struct {
	ChatCompletion bool `json:"chat_completion"`
	Completion     bool `json:"completion"`
	Embedding      bool `json:"embedding"`
	FineTuning     bool `json:"fine_tuning"`
	Vision         bool `json:"vision"`
}

// AzureRegion represents an Azure region.
type AzureRegion string

const (
	RegionEastUS      AzureRegion = "eastus"
	RegionEastUS2     AzureRegion = "eastus2"
	RegionWestUS      AzureRegion = "westus"
	RegionWestUS2     AzureRegion = "westus2"
	RegionWestUS3     AzureRegion = "westus3"
	RegionCentralUS   AzureRegion = "centralus"
	RegionNorthCentralUS AzureRegion = "northcentralus"
	RegionSouthCentralUS AzureRegion = "southcentralus"
	RegionWestCentralUS  AzureRegion = "westcentralus"
	RegionCanadaEast     AzureRegion = "canadaeast"
	RegionCanadaCentral  AzureRegion = "canadacentral"
	RegionBrazilSouth    AzureRegion = "brazilsouth"
	RegionNorthEurope    AzureRegion = "northeurope"
	RegionWestEurope     AzureRegion = "westeurope"
	RegionFranceCentral  AzureRegion = "francecentral"
	RegionGermanyWestCentral AzureRegion = "germanywestcentral"
	RegionNorwayEast        AzureRegion = "norwayeast"
	RegionSwitzerlandNorth  AzureRegion = "switzerlandnorth"
	RegionSwitzerlandWest   AzureRegion = "switzerlandwest"
	RegionUKSouth           AzureRegion = "uksouth"
	RegionUKWest            AzureRegion = "ukwest"
	RegionSoutheastAsia     AzureRegion = "southeastasia"
	RegionEastAsia          AzureRegion = "eastasia"
	RegionAustraliaEast     AzureRegion = "australiaeast"
	RegionAustraliaSoutheast AzureRegion = "australiasoutheast"
	RegionJapanEast          AzureRegion = "japaneast"
	RegionJapanWest          AzureRegion = "japanwest"
	RegionKoreaCentral       AzureRegion = "koreacentral"
	RegionKoreaSouth         AzureRegion = "koreasouth"
	RegionCentralIndia       AzureRegion = "centralindia"
	RegionSouthIndia         AzureRegion = "southindia"
	RegionWestIndia          AzureRegion = "westindia"
	RegionUAENorth           AzureRegion = "uaenorth"
	RegionSouthAfricaNorth   AzureRegion = "southafricanorth"
)
