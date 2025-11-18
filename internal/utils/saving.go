package utils

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

// ModuleMetadata contains metadata about a saved module.
type ModuleMetadata struct {
	Name              string                 `json:"name"`
	Version           string                 `json:"version"`
	SavedAt           time.Time              `json:"saved_at"`
	GoVersion         string                 `json:"go_version"`
	DependencyVersions map[string]string     `json:"dependency_versions"`
	CustomMetadata    map[string]interface{} `json:"custom_metadata,omitempty"`
}

// SaveOptions configures module saving.
type SaveOptions struct {
	// Path is the directory to save to
	Path string
	// Metadata is custom metadata to include
	Metadata map[string]interface{}
	// Version is the module version
	Version string
}

// Save saves a module with metadata to disk.
func Save(module interface{}, opts SaveOptions) error {
	if opts.Path == "" {
		return fmt.Errorf("save path is required")
	}

	// Create directory if it doesn't exist
	if err := os.MkdirAll(opts.Path, 0755); err != nil {
		return fmt.Errorf("creating directory: %w", err)
	}

	// Prepare metadata
	metadata := ModuleMetadata{
		Name:              fmt.Sprintf("%T", module),
		Version:           opts.Version,
		SavedAt:           time.Now(),
		GoVersion:         runtime.Version(),
		DependencyVersions: getDependencyVersions(),
		CustomMetadata:    opts.Metadata,
	}

	// Save metadata
	metadataPath := filepath.Join(opts.Path, "metadata.json")
	metadataFile, err := os.Create(metadataPath)
	if err != nil {
		return fmt.Errorf("creating metadata file: %w", err)
	}
	defer metadataFile.Close()

	encoder := json.NewEncoder(metadataFile)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(metadata); err != nil {
		return fmt.Errorf("encoding metadata: %w", err)
	}

	// Save module data
	modulePath := filepath.Join(opts.Path, "module.json")
	moduleFile, err := os.Create(modulePath)
	if err != nil {
		return fmt.Errorf("creating module file: %w", err)
	}
	defer moduleFile.Close()

	moduleEncoder := json.NewEncoder(moduleFile)
	moduleEncoder.SetIndent("", "  ")
	if err := moduleEncoder.Encode(module); err != nil {
		return fmt.Errorf("encoding module: %w", err)
	}

	return nil
}

// Load loads a module from disk.
func Load(path string, target interface{}) (*ModuleMetadata, error) {
	// Load metadata
	metadataPath := filepath.Join(path, "metadata.json")
	metadataFile, err := os.Open(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("opening metadata file: %w", err)
	}
	defer metadataFile.Close()

	var metadata ModuleMetadata
	if err := json.NewDecoder(metadataFile).Decode(&metadata); err != nil {
		return nil, fmt.Errorf("decoding metadata: %w", err)
	}

	// Check compatibility
	currentVersions := getDependencyVersions()
	for dep, savedVersion := range metadata.DependencyVersions {
		if currentVersion, ok := currentVersions[dep]; ok {
			if currentVersion != savedVersion {
				// Log warning but don't fail
				fmt.Printf("Warning: dependency %s version mismatch (saved: %s, current: %s)\n",
					dep, savedVersion, currentVersion)
			}
		}
	}

	// Load module data
	modulePath := filepath.Join(path, "module.json")
	moduleFile, err := os.Open(modulePath)
	if err != nil {
		return nil, fmt.Errorf("opening module file: %w", err)
	}
	defer moduleFile.Close()

	if err := json.NewDecoder(moduleFile).Decode(target); err != nil {
		return nil, fmt.Errorf("decoding module: %w", err)
	}

	return &metadata, nil
}

// getDependencyVersions returns the versions of key dependencies.
func getDependencyVersions() map[string]string {
	return map[string]string{
		"go": runtime.Version(),
		// Add other dependencies as needed
	}
}
