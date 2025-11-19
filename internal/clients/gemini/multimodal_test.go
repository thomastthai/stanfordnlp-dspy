package gemini

import (
	"bytes"
	"encoding/base64"
	"os"
	"path/filepath"
	"testing"
)

func TestNewTextPart(t *testing.T) {
	part := NewTextPart("Hello, world!")

	if part.Text != "Hello, world!" {
		t.Errorf("got text %q, want %q", part.Text, "Hello, world!")
	}

	if part.InlineData != nil {
		t.Error("InlineData should be nil for text part")
	}
}

func TestNewImagePartFromBase64(t *testing.T) {
	base64Data := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	mimeType := "image/png"

	part := NewImagePartFromBase64(base64Data, mimeType)

	if part.InlineData == nil {
		t.Fatal("InlineData should not be nil")
	}

	if part.InlineData.MimeType != mimeType {
		t.Errorf("got MIME type %q, want %q", part.InlineData.MimeType, mimeType)
	}

	if part.InlineData.Data != base64Data {
		t.Errorf("got data %q, want %q", part.InlineData.Data, base64Data)
	}
}

func TestNewImagePartFromReader(t *testing.T) {
	testData := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}
	reader := bytes.NewReader(testData)
	mimeType := "image/png"

	part, err := NewImagePartFromReader(reader, mimeType)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if part.InlineData == nil {
		t.Fatal("InlineData should not be nil")
	}

	expectedBase64 := base64.StdEncoding.EncodeToString(testData)
	if part.InlineData.Data != expectedBase64 {
		t.Errorf("got data %q, want %q", part.InlineData.Data, expectedBase64)
	}
}

func TestGetMIMETypeFromPath(t *testing.T) {
	tests := []struct {
		path     string
		expected string
	}{
		// Images
		{"image.jpg", "image/jpeg"},
		{"image.jpeg", "image/jpeg"},
		{"image.png", "image/png"},
		{"image.gif", "image/gif"},
		{"image.webp", "image/webp"},
		{"image.heic", "image/heic"},
		{"IMAGE.JPG", "image/jpeg"}, // Case insensitive

		// Videos
		{"video.mp4", "video/mp4"},
		{"video.mpeg", "video/mpeg"},
		{"video.mov", "video/mov"},
		{"video.avi", "video/avi"},
		{"video.webm", "video/webm"},

		// Audio
		{"audio.mp3", "audio/mp3"},
		{"audio.wav", "audio/wav"},
		{"audio.aac", "audio/aac"},

		// Unsupported
		{"file.txt", ""},
		{"file", ""},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			result := getMIMETypeFromPath(tt.path)
			if result != tt.expected {
				t.Errorf("got %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestNewImagePartFromFile(t *testing.T) {
	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "test.png")

	pngData := []byte{0x89, 0x50, 0x4e, 0x47}
	err := os.WriteFile(tmpFile, pngData, 0644)
	if err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	part, err := NewImagePartFromFile(tmpFile)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if part.InlineData == nil {
		t.Fatal("InlineData should not be nil")
	}

	if part.InlineData.MimeType != "image/png" {
		t.Errorf("got MIME type %q, want %q", part.InlineData.MimeType, "image/png")
	}

	// Test unsupported file type
	tmpFile2 := filepath.Join(tmpDir, "test.txt")
	err = os.WriteFile(tmpFile2, []byte("test"), 0644)
	if err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	_, err = NewImagePartFromFile(tmpFile2)
	if err == nil {
		t.Error("expected error for unsupported file type")
	}
}

func TestNewVideoPartFromURI(t *testing.T) {
	uri := "gs://my-bucket/video.mp4"
	mimeType := "video/mp4"

	part := NewVideoPartFromURI(uri, mimeType)

	if part.FileData == nil {
		t.Fatal("FileData should not be nil")
	}

	if part.FileData.FileURI != uri {
		t.Errorf("got URI %q, want %q", part.FileData.FileURI, uri)
	}

	if part.FileData.MimeType != mimeType {
		t.Errorf("got MIME type %q, want %q", part.FileData.MimeType, mimeType)
	}
}

func TestNewMultiModalContent(t *testing.T) {
	textPart := NewTextPart("Describe this")
	imagePart := NewImagePartFromBase64("base64data", "image/png")

	content := NewMultiModalContent("user", textPart, imagePart)

	if content.Role != "user" {
		t.Errorf("got role %q, want %q", content.Role, "user")
	}

	if len(content.Parts) != 2 {
		t.Errorf("got %d parts, want 2", len(content.Parts))
	}

	if content.Parts[0].Text != "Describe this" {
		t.Errorf("first part should be text")
	}

	if content.Parts[1].InlineData == nil {
		t.Error("second part should have InlineData")
	}
}

func TestSupportedFormats(t *testing.T) {
	t.Run("images", func(t *testing.T) {
		formats := SupportedImageFormats()
		if len(formats) == 0 {
			t.Error("expected at least one supported image format")
		}
		for _, format := range formats {
			if format == "" {
				t.Error("format should not be empty")
			}
		}
	})

	t.Run("videos", func(t *testing.T) {
		formats := SupportedVideoFormats()
		if len(formats) == 0 {
			t.Error("expected at least one supported video format")
		}
	})

	t.Run("audio", func(t *testing.T) {
		formats := SupportedAudioFormats()
		if len(formats) == 0 {
			t.Error("expected at least one supported audio format")
		}
	})
}

func TestValidateFileSize(t *testing.T) {
	tests := []struct {
		name    string
		size    int
		wantErr bool
	}{
		{"small file", 1024, false},
		{"medium file", 10 * 1024 * 1024, false},
		{"max size", MaxFileSize, false},
		{"too large", MaxFileSize + 1, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateFileSize(tt.size)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateFileSize() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestIsVideoMIMEType(t *testing.T) {
	tests := []struct {
		mimeType string
		expected bool
	}{
		{"video/mp4", true},
		{"video/mpeg", true},
		{"image/jpeg", false},
		{"audio/mp3", false},
		{"text/plain", false},
	}

	for _, tt := range tests {
		t.Run(tt.mimeType, func(t *testing.T) {
			result := isVideoMIMEType(tt.mimeType)
			if result != tt.expected {
				t.Errorf("got %v, want %v", result, tt.expected)
			}
		})
	}
}
