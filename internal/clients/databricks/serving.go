package databricks

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/hashicorp/go-retryablehttp"
)

// EndpointInfo represents information about a Databricks serving endpoint.
type EndpointInfo struct {
	ID           string          `json:"id"`
	Name         string          `json:"name"`
	Creator      string          `json:"creator"`
	CreationTime int64           `json:"creation_timestamp"`
	LastUpdated  int64           `json:"last_updated_timestamp"`
	State        EndpointState   `json:"state"`
	Config       *EndpointConfig `json:"config,omitempty"`
	Tags         []Tag           `json:"tags,omitempty"`
}

// EndpointState represents the state of a serving endpoint.
type EndpointState struct {
	Ready        string `json:"ready"`
	ConfigUpdate string `json:"config_update,omitempty"`
}

// EndpointConfig represents the configuration of a serving endpoint.
type EndpointConfig struct {
	ServedModels  []ServedModel  `json:"served_models"`
	TrafficConfig *TrafficConfig `json:"traffic_config,omitempty"`
}

// ServedModel represents a model served by an endpoint.
type ServedModel struct {
	Name               string            `json:"name"`
	ModelName          string            `json:"model_name"`
	ModelVersion       string            `json:"model_version"`
	WorkloadSize       string            `json:"workload_size"`
	ScaleToZeroEnabled bool              `json:"scale_to_zero_enabled"`
	EnvironmentVars    map[string]string `json:"environment_vars,omitempty"`
}

// TrafficConfig defines how traffic is routed between models.
type TrafficConfig struct {
	Routes []TrafficRoute `json:"routes"`
}

// TrafficRoute defines a traffic routing rule.
type TrafficRoute struct {
	ServedModelName   string `json:"served_model_name"`
	TrafficPercentage int    `json:"traffic_percentage"`
}

// Tag represents a key-value tag on an endpoint.
type Tag struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// ListEndpoints lists all serving endpoints in the workspace.
func (c *Client) ListEndpoints(ctx context.Context) ([]EndpointInfo, error) {
	url := fmt.Sprintf("%s/api/2.0/serving-endpoints", c.host)

	// Create HTTP request
	httpReq, err := retryablehttp.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.token))

	// Make the API call
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Databricks API call failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Databricks API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	type ListResponse struct {
		Endpoints []EndpointInfo `json:"endpoints"`
	}

	var listResp ListResponse
	if err := json.Unmarshal(body, &listResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return listResp.Endpoints, nil
}

// GetEndpoint gets information about a specific serving endpoint.
func (c *Client) GetEndpoint(ctx context.Context, name string) (*EndpointInfo, error) {
	url := fmt.Sprintf("%s/api/2.0/serving-endpoints/%s", c.host, name)

	// Create HTTP request
	httpReq, err := retryablehttp.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.token))

	// Make the API call
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Databricks API call failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Databricks API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var endpointInfo EndpointInfo
	if err := json.Unmarshal(body, &endpointInfo); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &endpointInfo, nil
}

// EndpointMetrics represents metrics for a serving endpoint.
type EndpointMetrics struct {
	Key       string  `json:"key"`
	Value     float64 `json:"value,omitempty"`
	Timestamp int64   `json:"timestamp,omitempty"`
}

// GetEndpointMetrics retrieves metrics for a serving endpoint.
func (c *Client) GetEndpointMetrics(ctx context.Context, name string) ([]EndpointMetrics, error) {
	url := fmt.Sprintf("%s/api/2.0/serving-endpoints/%s/metrics", c.host, name)

	// Create HTTP request
	httpReq, err := retryablehttp.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.token))

	// Make the API call
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Databricks API call failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Check for errors
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Databricks API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	type MetricsResponse struct {
		Metrics []EndpointMetrics `json:"metrics"`
	}

	var metricsResp MetricsResponse
	if err := json.Unmarshal(body, &metricsResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return metricsResp.Metrics, nil
}
