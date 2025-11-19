package gemini

import (
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"strings"
)

// InlineData represents inline data (e.g., images, audio) in base64 format.
type InlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64 encoded
}

// FileData represents a reference to a file in Google Cloud Storage or other sources.
type FileData struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

// VideoPart represents a video part in the content.
type VideoPart struct {
	VideoMetadata *VideoMetadata `json:"videoMetadata,omitempty"`
}

// VideoMetadata contains metadata about a video.
type VideoMetadata struct {
	StartOffsetSec float64 `json:"startOffsetSec,omitempty"`
	EndOffsetSec   float64 `json:"endOffsetSec,omitempty"`
}

// MultiModalPart represents a part that can be text, image, audio, or video.
type MultiModalPart struct {
	Text       string      `json:"text,omitempty"`
	InlineData *InlineData `json:"inlineData,omitempty"`
	FileData   *FileData   `json:"fileData,omitempty"`
}

// NewTextPart creates a text part.
func NewTextPart(text string) MultiModalPart {
	return MultiModalPart{
		Text: text,
	}
}

// NewImagePartFromBase64 creates an image part from base64 data.
func NewImagePartFromBase64(base64Data, mimeType string) MultiModalPart {
	return MultiModalPart{
		InlineData: &InlineData{
			MimeType: mimeType,
			Data:     base64Data,
		},
	}
}

// NewImagePartFromFile creates an image part from a file.
func NewImagePartFromFile(filePath string) (MultiModalPart, error) {
	// Read file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return MultiModalPart{}, fmt.Errorf("failed to read file: %w", err)
	}

	// Determine MIME type from extension
	mimeType := getMIMETypeFromPath(filePath)
	if mimeType == "" {
		return MultiModalPart{}, fmt.Errorf("unsupported file type: %s", filePath)
	}

	// Encode to base64
	base64Data := base64.StdEncoding.EncodeToString(data)

	return NewImagePartFromBase64(base64Data, mimeType), nil
}

// NewImagePartFromReader creates an image part from an io.Reader.
func NewImagePartFromReader(reader io.Reader, mimeType string) (MultiModalPart, error) {
	// Read all data
	data, err := io.ReadAll(reader)
	if err != nil {
		return MultiModalPart{}, fmt.Errorf("failed to read data: %w", err)
	}

	// Encode to base64
	base64Data := base64.StdEncoding.EncodeToString(data)

	return NewImagePartFromBase64(base64Data, mimeType), nil
}

// NewVideoPartFromFile creates a video part from a file.
func NewVideoPartFromFile(filePath string) (MultiModalPart, error) {
	// Read file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return MultiModalPart{}, fmt.Errorf("failed to read file: %w", err)
	}

	// Determine MIME type from extension
	mimeType := getMIMETypeFromPath(filePath)
	if mimeType == "" || !isVideoMIMEType(mimeType) {
		return MultiModalPart{}, fmt.Errorf("unsupported video type: %s", filePath)
	}

	// Encode to base64
	base64Data := base64.StdEncoding.EncodeToString(data)

	return MultiModalPart{
		InlineData: &InlineData{
			MimeType: mimeType,
			Data:     base64Data,
		},
	}, nil
}

// NewVideoPartFromURI creates a video part from a Google Cloud Storage URI.
func NewVideoPartFromURI(uri, mimeType string) MultiModalPart {
	return MultiModalPart{
		FileData: &FileData{
			MimeType: mimeType,
			FileURI:  uri,
		},
	}
}

// NewAudioPartFromBase64 creates an audio part from base64 data.
func NewAudioPartFromBase64(base64Data, mimeType string) MultiModalPart {
	return MultiModalPart{
		InlineData: &InlineData{
			MimeType: mimeType,
			Data:     base64Data,
		},
	}
}

// getMIMETypeFromPath determines the MIME type from a file path.
func getMIMETypeFromPath(path string) string {
	path = strings.ToLower(path)

	switch {
	// Images
	case strings.HasSuffix(path, ".jpg"), strings.HasSuffix(path, ".jpeg"):
		return "image/jpeg"
	case strings.HasSuffix(path, ".png"):
		return "image/png"
	case strings.HasSuffix(path, ".gif"):
		return "image/gif"
	case strings.HasSuffix(path, ".webp"):
		return "image/webp"
	case strings.HasSuffix(path, ".heic"):
		return "image/heic"
	case strings.HasSuffix(path, ".heif"):
		return "image/heif"

	// Videos
	case strings.HasSuffix(path, ".mp4"):
		return "video/mp4"
	case strings.HasSuffix(path, ".mpeg"), strings.HasSuffix(path, ".mpg"):
		return "video/mpeg"
	case strings.HasSuffix(path, ".mov"):
		return "video/mov"
	case strings.HasSuffix(path, ".avi"):
		return "video/avi"
	case strings.HasSuffix(path, ".webm"):
		return "video/webm"
	case strings.HasSuffix(path, ".wmv"):
		return "video/x-ms-wmv"
	case strings.HasSuffix(path, ".3gp"):
		return "video/3gpp"

	// Audio
	case strings.HasSuffix(path, ".mp3"):
		return "audio/mp3"
	case strings.HasSuffix(path, ".wav"):
		return "audio/wav"
	case strings.HasSuffix(path, ".aac"):
		return "audio/aac"
	case strings.HasSuffix(path, ".ogg"):
		return "audio/ogg"
	case strings.HasSuffix(path, ".flac"):
		return "audio/flac"

	default:
		return ""
	}
}

// isVideoMIMEType checks if a MIME type is a video type.
func isVideoMIMEType(mimeType string) bool {
	return strings.HasPrefix(mimeType, "video/")
}

// SupportedImageFormats returns the list of supported image formats.
func SupportedImageFormats() []string {
	return []string{
		"image/jpeg",
		"image/png",
		"image/gif",
		"image/webp",
		"image/heic",
		"image/heif",
	}
}

// SupportedVideoFormats returns the list of supported video formats.
func SupportedVideoFormats() []string {
	return []string{
		"video/mp4",
		"video/mpeg",
		"video/mov",
		"video/avi",
		"video/webm",
		"video/x-ms-wmv",
		"video/3gpp",
	}
}

// SupportedAudioFormats returns the list of supported audio formats.
func SupportedAudioFormats() []string {
	return []string{
		"audio/mp3",
		"audio/wav",
		"audio/aac",
		"audio/ogg",
		"audio/flac",
	}
}

// MaxFileSize returns the maximum file size in bytes.
const MaxFileSize = 20 * 1024 * 1024 // 20MB

// ValidateFileSize checks if the file size is within limits.
func ValidateFileSize(size int) error {
	if size > MaxFileSize {
		return fmt.Errorf("file size %d bytes exceeds maximum of %d bytes", size, MaxFileSize)
	}
	return nil
}

// MultiModalContent represents content with multiple parts.
type MultiModalContent struct {
	Role  string           `json:"role"`
	Parts []MultiModalPart `json:"parts"`
}

// NewMultiModalContent creates a new multi-modal content.
func NewMultiModalContent(role string, parts ...MultiModalPart) MultiModalContent {
	return MultiModalContent{
		Role:  role,
		Parts: parts,
	}
}

// CreateVisionMessage creates a message with text and an image.
func CreateVisionMessage(text, imagePath string) (MultiModalContent, error) {
	imagePart, err := NewImagePartFromFile(imagePath)
	if err != nil {
		return MultiModalContent{}, err
	}

	return NewMultiModalContent(
		"user",
		NewTextPart(text),
		imagePart,
	), nil
}

// CreateVideoMessage creates a message with text and a video.
func CreateVideoMessage(text, videoPath string) (MultiModalContent, error) {
	videoPart, err := NewVideoPartFromFile(videoPath)
	if err != nil {
		return MultiModalContent{}, err
	}

	return NewMultiModalContent(
		"user",
		NewTextPart(text),
		videoPart,
	), nil
}
