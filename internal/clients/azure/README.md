# Azure OpenAI Client

This package provides a comprehensive client for Azure OpenAI API with support for chat completions, streaming, vision/multimodal capabilities, and flexible authentication.

## Features

- ✅ Chat completions with Azure OpenAI models
- ✅ Streaming support for real-time responses
- ✅ Vision/multimodal support (GPT-4 Vision)
- ✅ Multiple authentication methods (API key, Azure AD)
- ✅ Configurable API versions
- ✅ Automatic retries with exponential backoff
- ✅ Rate limiting integration
- ✅ Comprehensive error handling

## Installation

```bash
go get github.com/stanfordnlp/dspy
```

## Quick Start

### Basic Usage with API Key

```go
package main

import (
    "context"
    "fmt"
    "log"
    
    "github.com/stanfordnlp/dspy/internal/clients"
    "github.com/stanfordnlp/dspy/internal/clients/azure"
)

func main() {
    // Create Azure OpenAI client
    client, err := azure.NewClient(azure.ClientOptions{
        Endpoint:   "https://your-resource.openai.azure.com",
        APIKey:     "your-api-key",
        APIVersion: "2024-02-15-preview", // Optional, defaults to latest
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Create a request
    request := &clients.Request{
        Messages: []clients.Message{
            {Role: "user", Content: "Hello, how are you?"},
        },
        Temperature: 0.7,
        MaxTokens:   100,
    }
    
    // Call the API (deployment name is required for Azure)
    response, err := client.Call(context.Background(), request, "gpt-4")
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
    _ "github.com/stanfordnlp/dspy/internal/clients/azure" // Register provider
)

func main() {
    // Set environment variables
    os.Setenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com")
    os.Setenv("AZURE_OPENAI_API_KEY", "your-api-key")
    
    // Create LM through provider
    provider := clients.GetProvider("azure")
    lm, err := provider.Create(map[string]interface{}{
        "model": "gpt-4", // This is your deployment name
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Use the LM
    request := clients.NewRequest().
        WithMessages(clients.NewMessage("user", "What is Go?")).
        WithTemperature(0.7)
    
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
chunkCh, errCh := client.Stream(context.Background(), request, "gpt-4")

// Process chunks
for {
    select {
    case chunk, ok := <-chunkCh:
        if !ok {
            // Stream complete
            return
        }
        fmt.Print(chunk.Delta) // Print each chunk as it arrives
        
    case err := <-errCh:
        if err != nil {
            log.Fatal(err)
        }
    }
}
```

### Vision/Multimodal

Send images along with text for vision-enabled models:

```go
// Create vision message with text and image
visionMsg := azure.VisionMessage{
    Role: "user",
    Content: []azure.MessageContent{
        azure.NewTextContent("What's in this image?"),
        azure.NewImageURLContent("https://example.com/image.jpg", "high"),
    },
}

// Call with vision
response, err := client.CallWithVision(
    context.Background(),
    []azure.VisionMessage{visionMsg},
    "gpt-4-vision", // Use vision-enabled deployment
    &clients.Request{MaxTokens: 300},
)
```

Load image from file:

```go
// Load image from disk
base64Data, mimeType, err := azure.LoadImageAsBase64("/path/to/image.jpg")
if err != nil {
    log.Fatal(err)
}

// Create vision message with base64 image
visionMsg := azure.VisionMessage{
    Role: "user",
    Content: []azure.MessageContent{
        azure.NewTextContent("Describe this image"),
        azure.NewImageBase64Content(base64Data, mimeType, "high"),
    },
}
```

### Azure AD Authentication

Use Azure Active Directory for authentication:

```go
// Custom token provider
tokenProvider := func(ctx context.Context) (*azure.TokenCredential, error) {
    // Your Azure AD token acquisition logic
    // This could use Azure SDK's credential types
    token, expiresAt, err := getAzureADToken()
    if err != nil {
        return nil, err
    }
    
    return &azure.TokenCredential{
        Token:     token,
        ExpiresAt: expiresAt,
    }, nil
}

// Create Azure AD auth provider
authProvider := azure.NewAzureADAuth(tokenProvider)

// Create client with custom auth
client, err := azure.NewClientWithAuth(
    azure.ClientOptions{
        Endpoint:   "https://your-resource.openai.azure.com",
        APIVersion: "2024-02-15-preview",
    },
    authProvider,
)
```

## Configuration

### Environment Variables

The provider supports the following environment variables:

- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI resource endpoint (e.g., `https://my-resource.openai.azure.com`)
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key

### ClientOptions

```go
type ClientOptions struct {
    Endpoint   string        // Required: Azure OpenAI endpoint
    APIKey     string        // Required: API key (unless using custom auth)
    APIVersion string        // Optional: API version (default: "2024-02-15-preview")
    Timeout    time.Duration // Optional: Request timeout (default: 60s)
}
```

### Supported API Versions

- `2024-02-15-preview` (default)
- `2024-06-01`
- `2024-10-21`

## Supported Models

Common Azure OpenAI deployment names (note: actual names are customizable):

### GPT-4o Series
- `gpt-4o`
- `gpt-4o-mini`

### GPT-4 Series
- `gpt-4-turbo`
- `gpt-4`
- `gpt-4-32k`

### GPT-3.5 Series
- `gpt-35-turbo`
- `gpt-35-turbo-16k`

**Note**: In Azure OpenAI, you create deployments with custom names that point to specific model versions. Use your deployment name when calling the API.

## Error Handling

The client provides comprehensive error handling with retryable errors:

```go
response, err := client.Call(ctx, request, "gpt-4")
if err != nil {
    if clientErr, ok := err.(*clients.ClientError); ok {
        fmt.Printf("Error: %s (code: %s, retryable: %v)\n",
            clientErr.Message,
            clientErr.Type,
            clientErr.Retryable,
        )
        
        if clientErr.Retryable {
            // Retry logic
        }
    }
}
```

### Common Error Codes

- `rate_limit_exceeded`: Rate limit hit (retryable)
- `quota_exceeded`: Quota exceeded (retryable)
- `invalid_api_key`: Invalid authentication (not retryable)
- `model_not_ready`: Model/deployment not ready (retryable)
- `content_filter`: Content filtered by Azure's content safety (not retryable)

## Rate Limiting

The client automatically handles rate limiting with exponential backoff. Configure retry behavior:

```go
client.httpClient.RetryMax = 5
client.httpClient.RetryWaitMin = 2 * time.Second
client.httpClient.RetryWaitMax = 30 * time.Second
```

## Best Practices

1. **Deployment Names**: Use descriptive deployment names in Azure that indicate the model and version
2. **API Versions**: Pin to a specific API version for production stability
3. **Timeouts**: Set appropriate timeouts based on your use case
4. **Retry Logic**: Use built-in retry logic for transient errors
5. **Error Handling**: Always check for retryable errors and implement appropriate retry strategies
6. **Streaming**: Use streaming for long responses to improve user experience
7. **Vision**: Resize large images before sending to reduce costs and latency

## Examples

See the `examples/` directory for complete working examples:

- `examples/azure_basic/` - Basic chat completion
- `examples/azure_streaming/` - Streaming responses
- `examples/azure_vision/` - Vision/multimodal
- `examples/azure_auth/` - Azure AD authentication

## License

This project is licensed under the MIT License - see the LICENSE file for details.
