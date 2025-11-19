package openai

import (
	"encoding/base64"
	"fmt"
	"io"
	"os"
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

// NewImageContentFromFile creates an image content block from a file.
func NewImageContentFromFile(filePath string, detail string) (ImageURLContent, error) {
	// Read file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return ImageURLContent{}, fmt.Errorf("failed to read file: %w", err)
	}

	// Determine MIME type
	mimeType := getMIMETypeFromExtension(filePath)
	if mimeType == "" {
		return ImageURLContent{}, fmt.Errorf("unsupported image format: %s", filePath)
	}

	// Encode to base64
	base64Data := base64.StdEncoding.EncodeToString(data)

	return NewImageBase64Content(base64Data, mimeType, detail), nil
}

// NewImageContentFromReader creates an image content block from an io.Reader.
func NewImageContentFromReader(reader io.Reader, mimeType, detail string) (ImageURLContent, error) {
	// Read all data
	data, err := io.ReadAll(reader)
	if err != nil {
		return ImageURLContent{}, fmt.Errorf("failed to read data: %w", err)
	}

	// Encode to base64
	base64Data := base64.StdEncoding.EncodeToString(data)

	return NewImageBase64Content(base64Data, mimeType, detail), nil
}

// getMIMETypeFromExtension returns the MIME type for an image file extension.
func getMIMETypeFromExtension(filePath string) string {
	// Simple extension-based MIME type detection
	ext := filePath[len(filePath)-4:]
	switch ext {
	case ".png":
		return "image/png"
	case ".jpg", "jpeg":
		return "image/jpeg"
	case ".gif":
		return "image/gif"
	case "webp":
		return "image/webp"
	default:
		return ""
	}
}

// CreateVisionMessage creates a vision message with text and images.
func CreateVisionMessage(role, text string, images ...ImageURLContent) VisionMessage {
	content := make([]MessageContent, 0, len(images)+1)
	
	// Add text first
	content = append(content, NewTextContent(text))
	
	// Add images
	for _, img := range images {
		content = append(content, img)
	}
	
	return VisionMessage{
		Role:    role,
		Content: content,
	}
}

// SupportedImageFormats returns the list of supported image formats for GPT-4V.
func SupportedImageFormats() []string {
	return []string{"image/png", "image/jpeg", "image/gif", "image/webp"}
}

// MaxImageSize returns the maximum image size in bytes (20MB for GPT-4V).
const MaxImageSize = 20 * 1024 * 1024

// ValidateImageSize checks if the image size is within limits.
func ValidateImageSize(size int) error {
	if size > MaxImageSize {
		return fmt.Errorf("image size %d bytes exceeds maximum of %d bytes", size, MaxImageSize)
	}
	return nil
}

// ImageDetail constants
const (
	ImageDetailLow  = "low"
	ImageDetailHigh = "high"
	ImageDetailAuto = "auto"
)
