package openai

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

func TestNewImageURLContent(t *testing.T) {
	url := "https://example.com/image.jpg"
	detail := "high"

	content := NewImageURLContent(url, detail)

	if content.Type != "image_url" {
		t.Errorf("got type %q, want %q", content.Type, "image_url")
	}

	if content.ImageURL.URL != url {
		t.Errorf("got URL %q, want %q", content.ImageURL.URL, url)
	}

	if content.ImageURL.Detail != detail {
		t.Errorf("got detail %q, want %q", content.ImageURL.Detail, detail)
	}
}

func TestNewImageURLContentDefaultDetail(t *testing.T) {
	url := "https://example.com/image.jpg"

	content := NewImageURLContent(url, "")

	if content.ImageURL.Detail != "auto" {
		t.Errorf("got detail %q, want %q", content.ImageURL.Detail, "auto")
	}
}

func TestNewImageBase64Content(t *testing.T) {
	base64Data := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	mimeType := "image/png"
	detail := "low"

	content := NewImageBase64Content(base64Data, mimeType, detail)

	if content.Type != "image_url" {
		t.Errorf("got type %q, want %q", content.Type, "image_url")
	}

	expectedURL := "data:image/png;base64," + base64Data
	if content.ImageURL.URL != expectedURL {
		t.Errorf("got URL %q, want %q", content.ImageURL.URL, expectedURL)
	}

	if content.ImageURL.Detail != detail {
		t.Errorf("got detail %q, want %q", content.ImageURL.Detail, detail)
	}
}

func TestNewImageContentFromReader(t *testing.T) {
	testData := []byte{0x89, 0x50, 0x4e, 0x47}
	reader := bytes.NewReader(testData)
	mimeType := "image/png"
	detail := "high"

	content, err := NewImageContentFromReader(reader, mimeType, detail)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if content.Type != "image_url" {
		t.Errorf("got type %q, want %q", content.Type, "image_url")
	}

	expectedBase64 := base64.StdEncoding.EncodeToString(testData)
	expectedURL := "data:image/png;base64," + expectedBase64
	if content.ImageURL.URL != expectedURL {
		t.Errorf("got URL %q, want %q", content.ImageURL.URL, expectedURL)
	}
}

func TestNewImageContentFromFile(t *testing.T) {
	tmpDir := t.TempDir()
	tmpFile := filepath.Join(tmpDir, "test.png")

	pngData := []byte{0x89, 0x50, 0x4e, 0x47}
	err := os.WriteFile(tmpFile, pngData, 0644)
	if err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	content, err := NewImageContentFromFile(tmpFile, "auto")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if content.Type != "image_url" {
		t.Errorf("got type %q, want %q", content.Type, "image_url")
	}

	// Test unsupported file type
	tmpFile2 := filepath.Join(tmpDir, "test.txt")
	err = os.WriteFile(tmpFile2, []byte("test"), 0644)
	if err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	_, err = NewImageContentFromFile(tmpFile2, "auto")
	if err == nil {
		t.Error("expected error for unsupported file type")
	}
}

func TestCreateVisionMessage(t *testing.T) {
	text := "What's in this image?"
	img1 := NewImageURLContent("https://example.com/img1.jpg", "high")
	img2 := NewImageURLContent("https://example.com/img2.jpg", "low")

	msg := CreateVisionMessage("user", text, img1, img2)

	if msg.Role != "user" {
		t.Errorf("got role %q, want %q", msg.Role, "user")
	}

	if len(msg.Content) != 3 {
		t.Fatalf("got %d content items, want 3", len(msg.Content))
	}

	// First should be text
	textContent, ok := msg.Content[0].(TextContent)
	if !ok {
		t.Fatal("first content should be TextContent")
	}
	if textContent.Text != text {
		t.Errorf("got text %q, want %q", textContent.Text, text)
	}

	// Second and third should be images
	imgContent1, ok := msg.Content[1].(ImageURLContent)
	if !ok {
		t.Fatal("second content should be ImageURLContent")
	}
	if imgContent1.ImageURL.Detail != "high" {
		t.Errorf("got detail %q, want %q", imgContent1.ImageURL.Detail, "high")
	}

	imgContent2, ok := msg.Content[2].(ImageURLContent)
	if !ok {
		t.Fatal("third content should be ImageURLContent")
	}
	if imgContent2.ImageURL.Detail != "low" {
		t.Errorf("got detail %q, want %q", imgContent2.ImageURL.Detail, "low")
	}
}

func TestSupportedImageFormats(t *testing.T) {
	formats := SupportedImageFormats()

	expectedFormats := []string{"image/png", "image/jpeg", "image/gif", "image/webp"}

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
		{"medium image", 10 * 1024 * 1024, false},
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

func TestImageDetailConstants(t *testing.T) {
	if ImageDetailLow != "low" {
		t.Errorf("ImageDetailLow = %q, want %q", ImageDetailLow, "low")
	}

	if ImageDetailHigh != "high" {
		t.Errorf("ImageDetailHigh = %q, want %q", ImageDetailHigh, "high")
	}

	if ImageDetailAuto != "auto" {
		t.Errorf("ImageDetailAuto = %q, want %q", ImageDetailAuto, "auto")
	}
}
