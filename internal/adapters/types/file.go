package types

import (
	"encoding/base64"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// File represents a file that can be sent to an LM.
type File struct {
	// Path is the local file path
	Path string

	// URL is the file URL (if loading from URL)
	URL string

	// Data is the raw file data
	Data []byte

	// Name is the filename
	Name string

	// MimeType is the MIME type of the file
	MimeType string

	// Base64 is the base64-encoded file data
	Base64 string

	// Size is the file size in bytes
	Size int64

	// Extension is the file extension
	Extension string
}

// NewFileFromPath creates a File from a local path.
func NewFileFromPath(path string) (*File, error) {
	// Check if file exists
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("failed to stat file: %w", err)
	}

	if info.IsDir() {
		return nil, fmt.Errorf("path is a directory, not a file")
	}

	// Extract filename and extension
	name := filepath.Base(path)
	ext := filepath.Ext(path)

	return &File{
		Path:      path,
		Name:      name,
		Size:      info.Size(),
		Extension: ext,
	}, nil
}

// NewFileFromURL creates a File from a URL.
func NewFileFromURL(url string) *File {
	// Extract filename from URL
	parts := strings.Split(url, "/")
	name := parts[len(parts)-1]
	ext := filepath.Ext(name)

	return &File{
		URL:       url,
		Name:      name,
		Extension: ext,
	}
}

// NewFileFromBytes creates a File from raw bytes.
func NewFileFromBytes(data []byte, name string) *File {
	ext := filepath.Ext(name)
	base64Str := base64.StdEncoding.EncodeToString(data)

	return &File{
		Data:      data,
		Name:      name,
		Size:      int64(len(data)),
		Extension: ext,
		Base64:    base64Str,
	}
}

// Load loads the file data from path or URL.
func (f *File) Load() error {
	if f.Data != nil {
		return nil // Already loaded
	}

	// Load from local path
	if f.Path != "" {
		return f.loadFromPath()
	}

	// Load from URL
	if f.URL != "" {
		return f.loadFromURL()
	}

	return fmt.Errorf("no path, URL, or data available")
}

// loadFromPath loads file data from the local path.
func (f *File) loadFromPath() error {
	data, err := os.ReadFile(f.Path)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	f.Data = data
	f.Size = int64(len(data))

	// Generate base64
	f.Base64 = base64.StdEncoding.EncodeToString(data)

	// Detect MIME type
	if f.MimeType == "" {
		f.MimeType = detectMimeType(f.Extension, data)
	}

	return nil
}

// loadFromURL loads file data from a URL.
func (f *File) loadFromURL() error {
	resp, err := http.Get(f.URL)
	if err != nil {
		return fmt.Errorf("failed to fetch file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to fetch file: status %d", resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read file data: %w", err)
	}

	f.Data = data
	f.Size = int64(len(data))

	// Generate base64
	f.Base64 = base64.StdEncoding.EncodeToString(data)

	// Get MIME type from response or detect
	if f.MimeType == "" {
		contentType := resp.Header.Get("Content-Type")
		if contentType != "" {
			f.MimeType = contentType
		} else {
			f.MimeType = detectMimeType(f.Extension, data)
		}
	}

	return nil
}

// ToBase64DataURL returns a data URL with the base64-encoded file.
func (f *File) ToBase64DataURL() (string, error) {
	if f.Base64 == "" {
		if err := f.Load(); err != nil {
			return "", err
		}
	}

	if f.MimeType == "" {
		f.MimeType = detectMimeType(f.Extension, f.Data)
	}

	return fmt.Sprintf("data:%s;base64,%s", f.MimeType, f.Base64), nil
}

// Validate checks if the file size is within limits.
func (f *File) Validate(maxSizeMB int) error {
	if f.Data == nil {
		if err := f.Load(); err != nil {
			return err
		}
	}

	sizeMB := int(f.Size / (1024 * 1024))
	if sizeMB > maxSizeMB {
		return fmt.Errorf("file size %d MB exceeds limit of %d MB", sizeMB, maxSizeMB)
	}

	return nil
}

// IsText returns true if the file is a text file.
func (f *File) IsText() bool {
	if f.MimeType == "" && f.Data == nil {
		_ = f.Load()
	}

	if f.MimeType == "" {
		return false
	}

	return strings.HasPrefix(f.MimeType, "text/") ||
		strings.Contains(f.MimeType, "json") ||
		strings.Contains(f.MimeType, "xml") ||
		strings.Contains(f.MimeType, "javascript") ||
		strings.Contains(f.MimeType, "yaml")
}

// IsImage returns true if the file is an image.
func (f *File) IsImage() bool {
	if f.MimeType == "" && f.Data == nil {
		_ = f.Load()
	}

	return strings.HasPrefix(f.MimeType, "image/")
}

// IsAudio returns true if the file is audio.
func (f *File) IsAudio() bool {
	if f.MimeType == "" && f.Data == nil {
		_ = f.Load()
	}

	return strings.HasPrefix(f.MimeType, "audio/")
}

// IsVideo returns true if the file is video.
func (f *File) IsVideo() bool {
	if f.MimeType == "" && f.Data == nil {
		_ = f.Load()
	}

	return strings.HasPrefix(f.MimeType, "video/")
}

// detectMimeType detects the MIME type from extension and content.
func detectMimeType(extension string, data []byte) string {
	// Try to detect from extension first
	if extension != "" {
		mimeType := mime.TypeByExtension(extension)
		if mimeType != "" {
			return mimeType
		}
	}

	// Detect from content using http.DetectContentType
	if len(data) > 0 {
		return http.DetectContentType(data)
	}

	return "application/octet-stream"
}
