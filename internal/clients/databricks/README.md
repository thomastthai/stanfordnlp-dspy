# Databricks Model Serving Client

This package provides a comprehensive client for Databricks Foundation Model APIs and custom model serving endpoints with support for streaming and PAT authentication.

## Features

- ✅ Foundation Model APIs (DBRX, Llama, Mixtral, MPT)
- ✅ Custom model serving endpoints
- ✅ Streaming support for real-time responses
- ✅ PAT (Personal Access Token) authentication
- ✅ Chat completions API compatibility
- ✅ Endpoint management and monitoring
- ✅ Comprehensive error handling
- ✅ Rate limiting integration

## Installation

```bash
go get github.com/stanfordnlp/dspy
```

## Quick Start

### Basic Usage with Foundation Models

```go
package main

import (
    "context"
    "fmt"
    "log"
    
    "github.com/stanfordnlp/dspy/internal/clients"
    "github.com/stanfordnlp/dspy/internal/clients/databricks"
)

func main() {
    // Create Databricks client
    client, err := databricks.NewClient(databricks.ClientOptions{
        Host:  "https://your-workspace.cloud.databricks.com",
        Token: "your-pat-token",
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Create a request
    request := &clients.Request{
        Messages: []clients.Message{
            {Role: "user", Content: "What is Databricks?"},
        },
        Temperature: 0.7,
        MaxTokens:   500,
    }
    
    // Call DBRX Instruct
    response, err := client.Call(context.Background(), request, "databricks-dbrx-instruct")
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println(response.Choices[0].Message.Content)
}
```

### Using the Provider Pattern

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"
    
    "github.com/stanfordnlp/dspy/internal/clients"
    _ "github.com/stanfordnlp/dspy/internal/clients/databricks" // Register provider
)

func main() {
    // Set environment variables
    os.Setenv("DATABRICKS_HOST", "https://your-workspace.cloud.databricks.com")
    os.Setenv("DATABRICKS_TOKEN", "your-pat-token")
    
    // Create LM through provider
    provider := clients.GetProvider("databricks")
    lm, err := provider.Create(map[string]interface{}{
        "model": "databricks-dbrx-instruct",
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Use the LM
    request := clients.NewRequest().
        WithMessages(clients.NewMessage("user", "Explain machine learning")).
        WithTemperature(0.7).
        WithMaxTokens(1000)
    
    response, err := lm.Call(context.Background(), request)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println(response.Choices[0].Message.Content)
}
```

## Advanced Features

### Streaming

Stream responses for real-time interaction:

```go
// Enable streaming
chunkCh, errCh := client.Stream(context.Background(), request, "databricks-dbrx-instruct")

// Process chunks
for {
    select {
    case chunk, ok := <-chunkCh:
        if !ok {
            // Stream complete
            return
        }
        fmt.Print(chunk.Delta) // Print each chunk as it arrives
        if chunk.Done {
            fmt.Println("\n[Stream completed]")
        }
        
    case err := <-errCh:
        if err != nil {
            log.Fatal(err)
        }
    }
}
```

### Custom Serving Endpoints

Use your own deployed models:

```go
// Call a custom serving endpoint
response, err := client.Call(
    context.Background(),
    request,
    "my-custom-model-endpoint", // Your endpoint name
)
```

### Endpoint Management

List and inspect serving endpoints:

```go
// List all endpoints
endpoints, err := client.ListEndpoints(context.Background())
if err != nil {
    log.Fatal(err)
}

for _, endpoint := range endpoints {
    fmt.Printf("Endpoint: %s (State: %s)\n", endpoint.Name, endpoint.State.Ready)
}

// Get specific endpoint info
endpointInfo, err := client.GetEndpoint(context.Background(), "databricks-dbrx-instruct")
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Model: %s\n", endpointInfo.Config.ServedModels[0].ModelName)
fmt.Printf("Version: %s\n", endpointInfo.Config.ServedModels[0].ModelVersion)
```

### Endpoint Metrics

Monitor endpoint performance:

```go
// Get endpoint metrics
metrics, err := client.GetEndpointMetrics(context.Background(), "databricks-dbrx-instruct")
if err != nil {
    log.Fatal(err)
}

for _, metric := range metrics {
    fmt.Printf("%s: %f\n", metric.Key, metric.Value)
}
```

## Supported Models

### Foundation Models

Databricks provides several pre-deployed foundation models:

#### DBRX (Databricks)
```go
"databricks-dbrx-instruct"  // DBRX Instruct - 132B parameters
```
- **Best for**: General purpose, reasoning, coding
- **Context**: 32K tokens
- **Streaming**: ✅

#### Meta Llama
```go
"databricks-meta-llama-3-70b-instruct"  // Llama 3 70B Instruct
"databricks-meta-llama-3-8b-instruct"   // Llama 3 8B Instruct
"databricks-meta-llama-2-70b-chat"      // Llama 2 70B Chat
```
- **Best for**: Chat, instruction following
- **Context**: 8K tokens (Llama 3), 4K tokens (Llama 2)
- **Streaming**: ✅

#### Mixtral
```go
"databricks-mixtral-8x7b-instruct"  // Mixtral 8x7B Instruct
```
- **Best for**: Multilingual, code generation
- **Context**: 32K tokens
- **Streaming**: ✅

#### MPT (MosaicML)
```go
"databricks-mpt-7b-instruct"   // MPT 7B Instruct
"databricks-mpt-30b-instruct"  // MPT 30B Instruct
```
- **Best for**: Instruction following, summarization
- **Context**: 8K tokens
- **Streaming**: ✅

### Custom Models

You can deploy your own models as serving endpoints:

```python
# Deploy a custom model (Python example)
from databricks import workspace

workspace.serving.create(
    name="my-custom-model",
    config={
        "served_models": [{
            "model_name": "my_model",
            "model_version": "1",
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
        }]
    }
)
```

Then use it in Go:
```go
response, err := client.Call(ctx, request, "my-custom-model")
```

## Configuration

### ClientOptions

```go
type ClientOptions struct {
    Host    string        // Required: Databricks workspace URL
    Token   string        // Required: Personal Access Token (PAT)
    Timeout time.Duration // Optional: Request timeout (default: 60s)
}
```

### Environment Variables

The provider supports:

- `DATABRICKS_HOST`: Workspace URL (e.g., `https://your-workspace.cloud.databricks.com`)
- `DATABRICKS_TOKEN`: Personal Access Token

### Creating a Personal Access Token

1. Log into your Databricks workspace
2. Click your username → User Settings
3. Go to Access Tokens
4. Click "Generate New Token"
5. Set expiration and description
6. Copy the token (it's shown only once)

```bash
# Set environment variable
export DATABRICKS_TOKEN=dapi1234567890abcdef
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
```

## Model Capabilities

Query model capabilities:

```go
// Get capabilities for a foundation model
caps := databricks.GetModelCapabilities(databricks.ModelDBRXInstruct)

fmt.Printf("Chat: %v\n", caps.ChatCompletion)
fmt.Printf("Streaming: %v\n", caps.Streaming)
fmt.Printf("Completion: %v\n", caps.Completion)
```

## Error Handling

Handle Databricks-specific errors:

```go
response, err := client.Call(ctx, request, endpoint)
if err != nil {
    if dbErr, ok := err.(*databricks.DatabricksError); ok {
        fmt.Printf("Error: %s (code: %s)\n", dbErr.Message, dbErr.ErrorCode)
        
        if dbErr.IsRetryable() {
            // Implement retry logic
            log.Println("Retryable error, will retry...")
        }
    }
}
```

### Common Error Codes

- `RESOURCE_EXHAUSTED`: Rate limit exceeded (retryable)
- `TEMPORARILY_UNAVAILABLE`: Endpoint temporarily down (retryable)
- `INTERNAL_ERROR`: Internal server error (retryable)
- `UNAUTHENTICATED`: Invalid token (not retryable)
- `PERMISSION_DENIED`: Insufficient permissions (not retryable)
- `INVALID_ARGUMENT`: Invalid request parameters (not retryable)
- `NOT_FOUND`: Endpoint not found (not retryable)

## Workload Sizes

When deploying custom models, choose appropriate workload sizes:

```go
databricks.WorkloadSizeSmall   // 1-8 concurrent requests
databricks.WorkloadSizeMedium  // 8-32 concurrent requests  
databricks.WorkloadSizeLarge   // 32+ concurrent requests
```

## Best Practices

1. **Model Selection**:
   - Use DBRX for best performance and reasoning
   - Use Llama 3 8B for cost-effective inference
   - Use Mixtral for multilingual applications

2. **Token Management**:
   - Rotate PAT tokens regularly
   - Use workspace-level tokens for production
   - Store tokens securely (never commit to git)

3. **Endpoint Management**:
   - Enable scale-to-zero for infrequently used endpoints
   - Monitor endpoint metrics regularly
   - Use traffic splitting for A/B testing

4. **Streaming**:
   - Use streaming for long responses (>100 tokens)
   - Implement proper error handling for streams
   - Buffer chunks appropriately

5. **Rate Limiting**:
   - Implement exponential backoff for rate limits
   - Monitor endpoint quotas in workspace
   - Consider batch processing for high volume

6. **Cost Optimization**:
   - Use scale-to-zero for development endpoints
   - Choose appropriate workload sizes
   - Monitor DBU consumption

## Pricing

Databricks Model Serving pricing is based on:

- **Model size**: Larger models cost more per request
- **Workload size**: Compute resources allocated
- **Uptime**: Hours the endpoint is running
- **Requests**: Number of inference requests

Foundation Model APIs typically include:
- Pay-per-token pricing
- No provisioning costs
- Automatic scaling

Custom endpoints include:
- DBU-based pricing
- Optional scale-to-zero
- Dedicated compute resources

Check your Databricks workspace for current pricing.

## Workspace Requirements

### Prerequisites

- Databricks workspace (AWS, Azure, or GCP)
- Model Serving enabled
- Appropriate user permissions:
  - `Can Query` for inference
  - `Can Manage` for endpoint management

### Supported Clouds

- AWS
- Azure
- GCP

All clouds support the same API interface.

## Examples

Complete examples are available:

- `examples/databricks_basic/` - Basic foundation model usage
- `examples/databricks_streaming/` - Streaming responses
- `examples/databricks_custom/` - Custom model endpoints
- `examples/databricks_monitoring/` - Endpoint monitoring

## Troubleshooting

### Authentication Issues

```go
// Test authentication
endpoints, err := client.ListEndpoints(context.Background())
if err != nil {
    log.Printf("Authentication failed: %v", err)
    // Check token and host
}
```

### Endpoint Not Found

```go
// List available endpoints
endpoints, err := client.ListEndpoints(context.Background())
for _, ep := range endpoints {
    fmt.Println(ep.Name)
}
```

### Rate Limits

If hitting rate limits:
1. Check workspace quotas
2. Implement retry logic (built-in)
3. Consider workload size increase
4. Contact Databricks support for quota increase

### Endpoint State

Check if endpoint is ready:

```go
info, err := client.GetEndpoint(ctx, endpointName)
if err != nil {
    log.Fatal(err)
}

if info.State.Ready != "READY" {
    log.Printf("Endpoint not ready: %s", info.State.Ready)
}
```

## Integration with Unity Catalog

For models registered in Unity Catalog:

```python
# Register model in Unity Catalog
mlflow.register_model(
    "runs:/run-id/model",
    "catalog.schema.model_name"
)

# Create serving endpoint
workspace.serving.create(
    name="uc-model-endpoint",
    config={
        "served_models": [{
            "model_name": "catalog.schema.model_name",
            "model_version": "1",
            "workload_size": "Small",
        }]
    }
)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
