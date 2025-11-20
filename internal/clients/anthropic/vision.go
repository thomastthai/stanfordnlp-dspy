package anthropic

import (
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"strings"
)

// ContentBlock represents a content block that can be text or image.
type ContentBlock struct {
	Type   string       `json:"type"` // "text" or "image"
	Text   string       `json:"text,omitempty"`
	Source *ImageSource `json:"source,omitempty"`
}

// ImageSource represents an image source.
type ImageSource struct {
	Type      string `json:"type"`           // "base64" or "url"
	MediaType string `json:"media_type"`     // "image/jpeg", "image/png", "image/gif", "image/webp"
	Data      string `json:"data,omitempty"` // base64 encoded data
	URL       string `json:"url,omitempty"`  // image URL (not yet supported by Anthropic)
}

// MultiModalMessage represents a message with multiple content blocks.
type MultiModalMessage struct {
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content"`
}

// NewTextContent creates a text content block.
func NewTextContent(text string) ContentBlock {
	return ContentBlock{
		Type: "text",
		Text: text,
	}
}

// NewImageContentFromBase64 creates an image content block from base64 data.
func NewImageContentFromBase64(base64Data, mediaType string) ContentBlock {
	return ContentBlock{
		Type: "image",
		Source: &ImageSource{
			Type:      "base64",
			MediaType: mediaType,
			Data:      base64Data,
		},
	}
}

// NewImageContentFromFile creates an image content block from a file.
func NewImageContentFromFile(filePath string) (ContentBlock, error) {
	// Read file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return ContentBlock{}, fmt.Errorf("failed to read file: %w", err)
	}

	// Determine media type from extension
	mediaType := getMediaTypeFromPath(filePath)
	if mediaType == "" {
		return ContentBlock{}, fmt.Errorf("unsupported file type: %s", filePath)
	}

	// Encode to base64
	base64Data := base64.StdEncoding.EncodeToString(data)

	return NewImageContentFromBase64(base64Data, mediaType), nil
}

// NewImageContentFromReader creates an image content block from an io.Reader.
func NewImageContentFromReader(reader io.Reader, mediaType string) (ContentBlock, error) {
	// Read all data
	data, err := io.ReadAll(reader)
	if err != nil {
		return ContentBlock{}, fmt.Errorf("failed to read data: %w", err)
	}

	// Encode to base64
	base64Data := base64.StdEncoding.EncodeToString(data)

	return NewImageContentFromBase64(base64Data, mediaType), nil
}

// getMediaTypeFromPath determines the media type from a file path.
func getMediaTypeFromPath(path string) string {
	path = strings.ToLower(path)

	switch {
	case strings.HasSuffix(path, ".jpg"), strings.HasSuffix(path, ".jpeg"):
		return "image/jpeg"
	case strings.HasSuffix(path, ".png"):
		return "image/png"
	case strings.HasSuffix(path, ".gif"):
		return "image/gif"
	case strings.HasSuffix(path, ".webp"):
		return "image/webp"
	default:
		return ""
	}
}

// NewMultiModalMessage creates a message with multiple content blocks.
func NewMultiModalMessage(role string, contents ...ContentBlock) MultiModalMessage {
	return MultiModalMessage{
		Role:    role,
		Content: contents,
	}
}

// SupportedImageFormats returns the list of supported image formats.
func SupportedImageFormats() []string {
	return []string{"image/jpeg", "image/png", "image/gif", "image/webp"}
}

// MaxImageSize returns the maximum image size in bytes (5MB for Claude).
const MaxImageSize = 5 * 1024 * 1024

// ValidateImageSize checks if the image size is within limits.
func ValidateImageSize(size int) error {
	if size > MaxImageSize {
		return fmt.Errorf("image size %d bytes exceeds maximum of %d bytes", size, MaxImageSize)
	}
	return nil
}

// Example usage functions

// CreateVisionMessage creates a message with text and an image.
func CreateVisionMessage(text, imagePath string) (MultiModalMessage, error) {
	imageContent, err := NewImageContentFromFile(imagePath)
	if err != nil {
		return MultiModalMessage{}, err
	}

	return NewMultiModalMessage(
		"user",
		NewTextContent(text),
		imageContent,
	), nil
}

// CreateVisionMessageFromBase64 creates a message with text and a base64-encoded image.
func CreateVisionMessageFromBase64(text, base64Data, mediaType string) MultiModalMessage {
	return NewMultiModalMessage(
		"user",
		NewTextContent(text),
		NewImageContentFromBase64(base64Data, mediaType),
	)
}
