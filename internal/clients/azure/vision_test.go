package azure

import (
	"testing"
)

func TestNewTextContent(t *testing.T) {
	text := "Hello, world!"
	content := NewTextContent(text)

	if content.Type != "text" {
		t.Errorf("expected type 'text', got '%s'", content.Type)
	}
	if content.Text != text {
		t.Errorf("expected text '%s', got '%s'", text, content.Text)
	}
}

func TestNewImageURLContent(t *testing.T) {
	url := "https://example.com/image.jpg"
	detail := "high"

	content := NewImageURLContent(url, detail)

	if content.Type != "image_url" {
		t.Errorf("expected type 'image_url', got '%s'", content.Type)
	}
	if content.ImageURL.URL != url {
		t.Errorf("expected URL '%s', got '%s'", url, content.ImageURL.URL)
	}
	if content.ImageURL.Detail != detail {
		t.Errorf("expected detail '%s', got '%s'", detail, content.ImageURL.Detail)
	}
}

func TestNewImageURLContent_DefaultDetail(t *testing.T) {
	url := "https://example.com/image.jpg"
	content := NewImageURLContent(url, "")

	if content.ImageURL.Detail != "auto" {
		t.Errorf("expected default detail 'auto', got '%s'", content.ImageURL.Detail)
	}
}

func TestNewImageBase64Content(t *testing.T) {
	base64Data := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	mimeType := "image/png"
	detail := "low"

	content := NewImageBase64Content(base64Data, mimeType, detail)

	expectedURL := "data:image/png;base64," + base64Data
	if content.ImageURL.URL != expectedURL {
		t.Errorf("expected URL '%s', got '%s'", expectedURL, content.ImageURL.URL)
	}
	if content.ImageURL.Detail != detail {
		t.Errorf("expected detail '%s', got '%s'", detail, content.ImageURL.Detail)
	}
}

func TestLoadImageAsBase64_NonExistentFile(t *testing.T) {
	_, _, err := LoadImageAsBase64("/nonexistent/file.jpg")
	if err == nil {
		t.Error("expected error for non-existent file")
	}
}
