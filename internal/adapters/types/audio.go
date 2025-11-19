package types

import (
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
)

// AudioFormat represents supported audio formats.
type AudioFormat string

const (
	AudioFormatMP3  AudioFormat = "mp3"
	AudioFormatWAV  AudioFormat = "wav"
	AudioFormatOGG  AudioFormat = "ogg"
	AudioFormatFLAC AudioFormat = "flac"
	AudioFormatM4A  AudioFormat = "m4a"
)

// Audio represents an audio file that can be sent to a multimodal LM.
type Audio struct {
	// URL is the audio URL (if loading from URL)
	URL string

	// Data is the raw audio data (if loading from bytes)
	Data []byte

	// Format is the audio format
	Format AudioFormat

	// Base64 is the base64-encoded audio data
	Base64 string

	// MimeType is the MIME type of the audio
	MimeType string

	// TranscriptionText is the transcription of the audio (if available)
	TranscriptionText string

	// DurationSeconds is the duration in seconds (if available)
	DurationSeconds float64
}

// NewAudioFromURL creates an Audio from a URL.
func NewAudioFromURL(url string) *Audio {
	return &Audio{
		URL: url,
	}
}

// NewAudioFromBytes creates an Audio from raw bytes.
func NewAudioFromBytes(data []byte, format AudioFormat) (*Audio, error) {
	mimeType, err := audioFormatToMimeType(format)
	if err != nil {
		return nil, err
	}

	base64Str := base64.StdEncoding.EncodeToString(data)

	return &Audio{
		Data:     data,
		Format:   format,
		Base64:   base64Str,
		MimeType: mimeType,
	}, nil
}

// Load fetches the audio data from URL if not already loaded.
func (a *Audio) Load() error {
	if a.Data != nil {
		return nil // Already loaded
	}

	if a.URL == "" {
		return fmt.Errorf("no URL or data available")
	}

	// Fetch from URL
	resp, err := http.Get(a.URL)
	if err != nil {
		return fmt.Errorf("failed to fetch audio: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to fetch audio: status %d", resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read audio data: %w", err)
	}

	a.Data = data

	// Detect format if not set
	if a.Format == "" {
		a.Format = detectAudioFormat(resp.Header.Get("Content-Type"))
	}

	// Generate base64
	a.Base64 = base64.StdEncoding.EncodeToString(data)

	// Set MIME type
	if a.MimeType == "" {
		mimeType, err := audioFormatToMimeType(a.Format)
		if err != nil {
			return err
		}
		a.MimeType = mimeType
	}

	return nil
}

// ToBase64DataURL returns a data URL with the base64-encoded audio.
func (a *Audio) ToBase64DataURL() (string, error) {
	if a.Base64 == "" {
		if err := a.Load(); err != nil {
			return "", err
		}
	}

	if a.MimeType == "" {
		mimeType, err := audioFormatToMimeType(a.Format)
		if err != nil {
			return "", err
		}
		a.MimeType = mimeType
	}

	return fmt.Sprintf("data:%s;base64,%s", a.MimeType, a.Base64), nil
}

// ToMessageContent converts the audio to a message content format for LM APIs.
func (a *Audio) ToMessageContent() (map[string]interface{}, error) {
	if a.URL != "" && a.Data == nil {
		// Use URL directly
		return map[string]interface{}{
			"type": "audio_url",
			"audio_url": map[string]string{
				"url": a.URL,
			},
		}, nil
	}

	// Use base64 data URL
	dataURL, err := a.ToBase64DataURL()
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"type": "audio_url",
		"audio_url": map[string]string{
			"url": dataURL,
		},
	}, nil
}

// Validate checks if the audio size is within limits.
func (a *Audio) Validate(maxSizeMB int) error {
	if a.Data == nil {
		if err := a.Load(); err != nil {
			return err
		}
	}

	sizeMB := len(a.Data) / (1024 * 1024)
	if sizeMB > maxSizeMB {
		return fmt.Errorf("audio size %d MB exceeds limit of %d MB", sizeMB, maxSizeMB)
	}

	return nil
}

// detectAudioFormat detects the audio format from the content type.
func detectAudioFormat(contentType string) AudioFormat {
	switch contentType {
	case "audio/mpeg", "audio/mp3":
		return AudioFormatMP3
	case "audio/wav", "audio/wave":
		return AudioFormatWAV
	case "audio/ogg":
		return AudioFormatOGG
	case "audio/flac":
		return AudioFormatFLAC
	case "audio/m4a", "audio/mp4":
		return AudioFormatM4A
	default:
		return ""
	}
}

// audioFormatToMimeType converts an AudioFormat to a MIME type.
func audioFormatToMimeType(format AudioFormat) (string, error) {
	switch format {
	case AudioFormatMP3:
		return "audio/mpeg", nil
	case AudioFormatWAV:
		return "audio/wav", nil
	case AudioFormatOGG:
		return "audio/ogg", nil
	case AudioFormatFLAC:
		return "audio/flac", nil
	case AudioFormatM4A:
		return "audio/m4a", nil
	default:
		return "", fmt.Errorf("unsupported format: %s", format)
	}
}
