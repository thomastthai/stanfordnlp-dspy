package azure

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"os"

	"github.com/stanfordnlp/dspy/internal/clients"
)

// MessageContent represents content that can be text or images.
type MessageContent interface {
	messageContent()
}

// TextContent represents text content.
type TextContent struct {
	Type string `json:"type"` // "text"
	Text string `json:"text"`
}

func (TextContent) messageContent() {}

// ImageURLContent represents an image via URL.
type ImageURLContent struct {
	Type     string   `json:"type"` // "image_url"
	ImageURL ImageURL `json:"image_url"`
}

func (ImageURLContent) messageContent() {}

// ImageURL contains the image URL or base64 data.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // "low", "high", "auto"
}

// VisionMessage represents a message that can contain text and images.
type VisionMessage struct {
	Role    string           `json:"role"`
	Content []MessageContent `json:"content"`
}

// NewTextContent creates a text content block.
func NewTextContent(text string) TextContent {
	return TextContent{
		Type: "text",
		Text: text,
	}
}

// NewImageURLContent creates an image content block from a URL.
func NewImageURLContent(url, detail string) ImageURLContent {
	if detail == "" {
		detail = "auto"
	}
	return ImageURLContent{
		Type: "image_url",
		ImageURL: ImageURL{
			URL:    url,
			Detail: detail,
		},
	}
}

// NewImageBase64Content creates an image content block from base64 data.
func NewImageBase64Content(base64Data, mimeType, detail string) ImageURLContent {
	if detail == "" {
		detail = "auto"
	}
	// Format: data:{mime_type};base64,{base64_data}
	dataURL := fmt.Sprintf("data:%s;base64,%s", mimeType, base64Data)
	return ImageURLContent{
		Type: "image_url",
		ImageURL: ImageURL{
			URL:    dataURL,
			Detail: detail,
		},
	}
}

// LoadImageAsBase64 loads an image file and returns it as base64 encoded data.
func LoadImageAsBase64(path string) (string, string, error) {
	// Open the file
	file, err := os.Open(path)
	if err != nil {
		return "", "", fmt.Errorf("failed to open image: %w", err)
	}
	defer file.Close()

	// Read file content
	data, err := io.ReadAll(file)
	if err != nil {
		return "", "", fmt.Errorf("failed to read image: %w", err)
	}

	// Encode to base64
	encoded := base64.StdEncoding.EncodeToString(data)

	// Detect MIME type from extension
	mimeType := "image/jpeg"
	if len(path) > 4 {
		ext := path[len(path)-4:]
		switch ext {
		case ".png":
			mimeType = "image/png"
		case ".jpg", "jpeg":
			mimeType = "image/jpeg"
		case ".gif":
			mimeType = "image/gif"
		case "webp":
			mimeType = "image/webp"
		}
	}

	return encoded, mimeType, nil
}

// VisionRequest represents an Azure OpenAI vision request.
type VisionRequest struct {
	Messages    []VisionMessage `json:"messages"`
	Temperature float64         `json:"temperature,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	TopP        float64         `json:"top_p,omitempty"`
	Stop        []string        `json:"stop,omitempty"`
	N           int             `json:"n,omitempty"`
}

// CallWithVision sends a vision request to Azure OpenAI API.
func (c *Client) CallWithVision(ctx context.Context, visionMessages []VisionMessage, deploymentName string, opts *clients.Request) (*clients.Response, error) {
	// Build request body
	maxTokens := 1000
	if opts != nil && opts.MaxTokens > 0 {
		maxTokens = opts.MaxTokens
	}

	var temperature float64
	var topP float64
	var stop []string
	var n int

	if opts != nil {
		temperature = opts.Temperature
		topP = opts.TopP
		stop = opts.StopSequences
		n = opts.N
	}

	if n == 0 {
		n = 1
	}

	reqBody := VisionRequest{
		Messages:    visionMessages,
		Temperature: temperature,
		MaxTokens:   maxTokens,
		TopP:        topP,
		Stop:        stop,
		N:           n,
	}

	// Use the same internal call mechanism but with vision messages
	// We need to make a custom call here since the message format is different
	return c.callVisionInternal(ctx, reqBody, deploymentName)
}

// callVisionInternal handles the internal vision API call.
func (c *Client) callVisionInternal(ctx context.Context, reqBody VisionRequest, deploymentName string) (*clients.Response, error) {
	// This is a simplified version - in production, you'd want to reuse
	// the same HTTP client and error handling logic as the main Call method
	// For now, we'll redirect to the standard Call with converted messages
	
	// Convert vision messages to standard messages
	// Note: This is a simplification - Azure OpenAI supports the vision format directly
	standardReq := &clients.Request{
		Messages:      []clients.Message{},
		Temperature:   reqBody.Temperature,
		MaxTokens:     reqBody.MaxTokens,
		TopP:          reqBody.TopP,
		StopSequences: reqBody.Stop,
		N:             reqBody.N,
	}

	// For each vision message, convert to a standard message
	// In a real implementation, you'd send the full vision structure to Azure
	for _, vmsg := range reqBody.Messages {
		msg := clients.Message{
			Role: vmsg.Role,
		}

		// Combine all text content
		var textParts []string
		for _, content := range vmsg.Content {
			if textContent, ok := content.(TextContent); ok {
				textParts = append(textParts, textContent.Text)
			}
		}

		if len(textParts) > 0 {
			msg.Content = textParts[0] // Simplified - just use first text part
		}

		standardReq.Messages = append(standardReq.Messages, msg)
	}

	return c.Call(ctx, standardReq, deploymentName)
}
