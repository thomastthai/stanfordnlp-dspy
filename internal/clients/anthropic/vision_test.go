package anthropic

import (
	"bytes"
	"encoding/base64"
	"os"
	"path/filepath"
	"testing"
)

func TestNewTextContent(t *testing.T) {
	content := NewTextContent("Hello, world!")

	if content.Type != "text" {
		t.Errorf("got type %q, want %q", content.Type, "text")
	}

	if content.Text != "Hello, world!" {
		t.Errorf("got text %q, want %q", content.Text, "Hello, world!")
	}
}

func TestNewImageContentFromBase64(t *testing.T) {
	base64Data := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	mediaType := "image/png"

	content := NewImageContentFromBase64(base64Data, mediaType)

	if content.Type != "image" {
		t.Errorf("got type %q, want %q", content.Type, "image")
	}

	if content.Source == nil {
		t.Fatal("source should not be nil")
	}

	if content.Source.Type != "base64" {
		t.Errorf("got source type %q, want %q", content.Source.Type, "base64")
	}

	if content.Source.MediaType != mediaType {
		t.Errorf("got media type %q, want %q", content.Source.MediaType, mediaType)
	}

	if content.Source.Data != base64Data {
		t.Errorf("got data %q, want %q", content.Source.Data, base64Data)
	}
}

func TestNewImageContentFromReader(t *testing.T) {
	// Create a small test image (1x1 PNG)
	testData := []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // PNG signature
	}

	reader := bytes.NewReader(testData)
	mediaType := "image/png"

	content, err := NewImageContentFromReader(reader, mediaType)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if content.Type != "image" {
		t.Errorf("got type %q, want %q", content.Type, "image")
	}

	if content.Source == nil {
		t.Fatal("source should not be nil")
	}

	expectedBase64 := base64.StdEncoding.EncodeToString(testData)
	if content.Source.Data != expectedBase64 {
		t.Errorf("got data %q, want %q", content.Source.Data, expectedBase64)
	}
}

func TestGetMediaTypeFromPath(t *testing.T) {
	tests := []struct {
		path     string
		expected string
	}{
		{"image.jpg", "image/jpeg"},
		{"image.jpeg", "image/jpeg"},
		{"image.png", "image/png"},
		{"image.gif", "image/gif"},
		{"image.webp", "image/webp"},
		{"IMAGE.JPG", "image/jpeg"}, // Case insensitive
		{"image.txt", ""},            // Unsupported
		{"image", ""},                // No extension
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			result := getMediaTypeFromPath(tt.path)
			if result != tt.expected {
				t.Errorf("got %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestNewImageContentFromFile(t *testing.T) {
	// Create a temporary PNG file
	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "test.png")

	// Small 1x1 PNG
	pngData := []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
		0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
	}

	err := os.WriteFile(tmpFile, pngData, 0644)
	if err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	content, err := NewImageContentFromFile(tmpFile)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if content.Type != "image" {
		t.Errorf("got type %q, want %q", content.Type, "image")
	}

	if content.Source.MediaType != "image/png" {
		t.Errorf("got media type %q, want %q", content.Source.MediaType, "image/png")
	}

	// Test unsupported file type
	tmpFile2 := filepath.Join(tmpDir, "test.txt")
	err = os.WriteFile(tmpFile2, []byte("test"), 0644)
	if err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	_, err = NewImageContentFromFile(tmpFile2)
	if err == nil {
		t.Error("expected error for unsupported file type")
	}
}

func TestNewMultiModalMessage(t *testing.T) {
	text := NewTextContent("Describe this image")
	image := NewImageContentFromBase64("base64data", "image/png")

	msg := NewMultiModalMessage("user", text, image)

	if msg.Role != "user" {
		t.Errorf("got role %q, want %q", msg.Role, "user")
	}

	if len(msg.Content) != 2 {
		t.Errorf("got %d content blocks, want 2", len(msg.Content))
	}

	if msg.Content[0].Type != "text" {
		t.Errorf("first content block should be text, got %q", msg.Content[0].Type)
	}

	if msg.Content[1].Type != "image" {
		t.Errorf("second content block should be image, got %q", msg.Content[1].Type)
	}
}

func TestSupportedImageFormats(t *testing.T) {
	formats := SupportedImageFormats()

	expectedFormats := []string{"image/jpeg", "image/png", "image/gif", "image/webp"}

	if len(formats) != len(expectedFormats) {
		t.Errorf("got %d formats, want %d", len(formats), len(expectedFormats))
	}

	for _, expected := range expectedFormats {
		found := false
		for _, format := range formats {
			if format == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected format %q not found", expected)
		}
	}
}

func TestValidateImageSize(t *testing.T) {
	tests := []struct {
		name    string
		size    int
		wantErr bool
	}{
		{"small image", 1024, false},
		{"medium image", 1024 * 1024, false},
		{"max size", MaxImageSize, false},
		{"too large", MaxImageSize + 1, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateImageSize(tt.size)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateImageSize() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCreateVisionMessageFromBase64(t *testing.T) {
	text := "What's in this image?"
	base64Data := "base64encodeddata"
	mediaType := "image/jpeg"

	msg := CreateVisionMessageFromBase64(text, base64Data, mediaType)

	if msg.Role != "user" {
		t.Errorf("got role %q, want %q", msg.Role, "user")
	}

	if len(msg.Content) != 2 {
		t.Fatalf("got %d content blocks, want 2", len(msg.Content))
	}

	if msg.Content[0].Text != text {
		t.Errorf("got text %q, want %q", msg.Content[0].Text, text)
	}

	if msg.Content[1].Source.Data != base64Data {
		t.Errorf("got data %q, want %q", msg.Content[1].Source.Data, base64Data)
	}
}
