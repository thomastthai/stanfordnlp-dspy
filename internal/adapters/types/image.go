// Package types provides special types for multimodal content.
package types

import (
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// ImageFormat represents supported image formats.
type ImageFormat string

const (
	ImageFormatJPEG ImageFormat = "jpeg"
	ImageFormatPNG  ImageFormat = "png"
	ImageFormatWebP ImageFormat = "webp"
	ImageFormatGIF  ImageFormat = "gif"
)

// Image represents an image that can be sent to a multimodal LM.
type Image struct {
	// URL is the image URL (if loading from URL)
	URL string

	// Data is the raw image data (if loading from bytes)
	Data []byte

	// Format is the image format
	Format ImageFormat

	// Base64 is the base64-encoded image data
	Base64 string

	// MimeType is the MIME type of the image
	MimeType string
}

// NewImageFromURL creates an Image from a URL.
func NewImageFromURL(url string) *Image {
	return &Image{
		URL: url,
	}
}

// NewImageFromBytes creates an Image from raw bytes.
func NewImageFromBytes(data []byte, format ImageFormat) (*Image, error) {
	mimeType, err := formatToMimeType(format)
	if err != nil {
		return nil, err
	}

	base64Str := base64.StdEncoding.EncodeToString(data)

	return &Image{
		Data:     data,
		Format:   format,
		Base64:   base64Str,
		MimeType: mimeType,
	}, nil
}

// NewImageFromBase64 creates an Image from a base64-encoded string.
func NewImageFromBase64(base64Str string, format ImageFormat) (*Image, error) {
	data, err := base64.StdEncoding.DecodeString(base64Str)
	if err != nil {
		return nil, fmt.Errorf("failed to decode base64: %w", err)
	}

	mimeType, err := formatToMimeType(format)
	if err != nil {
		return nil, err
	}

	return &Image{
		Data:     data,
		Format:   format,
		Base64:   base64Str,
		MimeType: mimeType,
	}, nil
}

// Load fetches the image data from URL if not already loaded.
func (img *Image) Load() error {
	if img.Data != nil {
		return nil // Already loaded
	}

	if img.URL == "" {
		return fmt.Errorf("no URL or data available")
	}

	// Fetch from URL
	resp, err := http.Get(img.URL)
	if err != nil {
		return fmt.Errorf("failed to fetch image: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to fetch image: status %d", resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read image data: %w", err)
	}

	img.Data = data

	// Detect format if not set
	if img.Format == "" {
		img.Format = detectFormat(data, resp.Header.Get("Content-Type"))
	}

	// Generate base64
	img.Base64 = base64.StdEncoding.EncodeToString(data)

	// Set MIME type
	if img.MimeType == "" {
		mimeType, err := formatToMimeType(img.Format)
		if err != nil {
			return err
		}
		img.MimeType = mimeType
	}

	return nil
}

// ToBase64DataURL returns a data URL with the base64-encoded image.
func (img *Image) ToBase64DataURL() (string, error) {
	if img.Base64 == "" {
		if err := img.Load(); err != nil {
			return "", err
		}
	}

	if img.MimeType == "" {
		mimeType, err := formatToMimeType(img.Format)
		if err != nil {
			return "", err
		}
		img.MimeType = mimeType
	}

	return fmt.Sprintf("data:%s;base64,%s", img.MimeType, img.Base64), nil
}

// ToMessageContent converts the image to a message content format for LM APIs.
func (img *Image) ToMessageContent() (map[string]interface{}, error) {
	if img.URL != "" && img.Data == nil {
		// Use URL directly
		return map[string]interface{}{
			"type": "image_url",
			"image_url": map[string]string{
				"url": img.URL,
			},
		}, nil
	}

	// Use base64 data URL
	dataURL, err := img.ToBase64DataURL()
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"type": "image_url",
		"image_url": map[string]string{
			"url": dataURL,
		},
	}, nil
}

// Validate checks if the image size is within limits.
func (img *Image) Validate(maxSizeMB int) error {
	if img.Data == nil {
		if err := img.Load(); err != nil {
			return err
		}
	}

	sizeMB := len(img.Data) / (1024 * 1024)
	if sizeMB > maxSizeMB {
		return fmt.Errorf("image size %d MB exceeds limit of %d MB", sizeMB, maxSizeMB)
	}

	return nil
}

// detectFormat detects the image format from the data and content type.
func detectFormat(data []byte, contentType string) ImageFormat {
	// Try content type first
	switch strings.ToLower(contentType) {
	case "image/jpeg", "image/jpg":
		return ImageFormatJPEG
	case "image/png":
		return ImageFormatPNG
	case "image/webp":
		return ImageFormatWebP
	case "image/gif":
		return ImageFormatGIF
	}

	// Try magic bytes
	if len(data) < 8 {
		return ""
	}

	// JPEG: FF D8 FF
	if data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
		return ImageFormatJPEG
	}

	// PNG: 89 50 4E 47 0D 0A 1A 0A
	if data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47 {
		return ImageFormatPNG
	}

	// WebP: RIFF .... WEBP
	if len(data) >= 12 && data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F' &&
		data[8] == 'W' && data[9] == 'E' && data[10] == 'B' && data[11] == 'P' {
		return ImageFormatWebP
	}

	// GIF: GIF87a or GIF89a
	if len(data) >= 6 && data[0] == 'G' && data[1] == 'I' && data[2] == 'F' {
		return ImageFormatGIF
	}

	return ""
}

// formatToMimeType converts an ImageFormat to a MIME type.
func formatToMimeType(format ImageFormat) (string, error) {
	switch format {
	case ImageFormatJPEG:
		return "image/jpeg", nil
	case ImageFormatPNG:
		return "image/png", nil
	case ImageFormatWebP:
		return "image/webp", nil
	case ImageFormatGIF:
		return "image/gif", nil
	default:
		return "", fmt.Errorf("unsupported format: %s", format)
	}
}
