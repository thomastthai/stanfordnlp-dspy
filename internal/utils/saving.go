package utils

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// SaveMetadata contains metadata about a saved model/module.
type SaveMetadata struct {
	Version     string                 `json:"version"`
	Timestamp   time.Time              `json:"timestamp"`
	Framework   string                 `json:"framework"`
	DSPyVersion string                 `json:"dspy_version"`
	GoVersion   string                 `json:"go_version"`
	Custom      map[string]interface{} `json:"custom,omitempty"`
}

// Saveable represents an object that can be saved and loaded.
type Saveable interface {
	// Save serializes the object to a map
	Save() (map[string]interface{}, error)
	
	// Load deserializes the object from a map
	Load(map[string]interface{}) error
}

// SaveOptions configures save behavior.
type SaveOptions struct {
	Compress       bool
	IncludeVersion bool
	Metadata       map[string]interface{}
}

// DefaultSaveOptions returns default save options.
func DefaultSaveOptions() SaveOptions {
	return SaveOptions{
		Compress:       false,
		IncludeVersion: true,
		Metadata:       make(map[string]interface{}),
	}
}

// Save saves a saveable object to a file with metadata.
func Save(obj Saveable, path string, opts SaveOptions) error {
	data, err := obj.Save()
	if err != nil {
		return fmt.Errorf("failed to serialize object: %w", err)
	}
	
	// Create save package with metadata
	savePackage := map[string]interface{}{
		"data": data,
	}
	
	if opts.IncludeVersion {
		metadata := SaveMetadata{
			Version:     "1.0",
			Timestamp:   time.Now(),
			Framework:   "dspy-go",
			DSPyVersion: "3.1.0",
			GoVersion:   "1.24",
			Custom:      opts.Metadata,
		}
		savePackage["metadata"] = metadata
	}
	
	// Serialize to JSON
	jsonData, err := json.MarshalIndent(savePackage, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal to JSON: %w", err)
	}
	
	// Write to file
	if err := os.WriteFile(path, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}
	
	return nil
}

// Load loads a saveable object from a file.
func Load(obj Saveable, path string) error {
	// Read file
	jsonData, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}
	
	// Parse JSON
	var savePackage map[string]interface{}
	if err := json.Unmarshal(jsonData, &savePackage); err != nil {
		return fmt.Errorf("failed to unmarshal JSON: %w", err)
	}
	
	// Extract data
	data, ok := savePackage["data"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid save format: missing or invalid data field")
	}
	
	// Load object
	if err := obj.Load(data); err != nil {
		return fmt.Errorf("failed to deserialize object: %w", err)
	}
	
	return nil
}

// LoadMetadata loads only the metadata from a saved file.
func LoadMetadata(path string) (*SaveMetadata, error) {
	jsonData, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}
	
	var savePackage map[string]interface{}
	if err := json.Unmarshal(jsonData, &savePackage); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}
	
	metadataMap, ok := savePackage["metadata"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("no metadata found")
	}
	
	// Convert to SaveMetadata
	metadataJSON, err := json.Marshal(metadataMap)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal metadata: %w", err)
	}
	
	var metadata SaveMetadata
	if err := json.Unmarshal(metadataJSON, &metadata); err != nil {
		return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
	}
	
	return &metadata, nil
}

// CheckCompatibility checks if a saved file is compatible with the current version.
func CheckCompatibility(path string, requiredVersion string) (bool, error) {
	metadata, err := LoadMetadata(path)
	if err != nil {
		return false, err
	}
	
	// Simple version check (in production, use semantic versioning)
	return metadata.Version == requiredVersion, nil
}

// MigrateIfNeeded migrates a saved file to the current version if needed.
func MigrateIfNeeded(path string, migrators map[string]func(map[string]interface{}) error) error {
	metadata, err := LoadMetadata(path)
	if err != nil {
		return fmt.Errorf("failed to load metadata: %w", err)
	}
	
	migrator, ok := migrators[metadata.Version]
	if !ok {
		// No migration needed or available
		return nil
	}
	
	// Read file
	jsonData, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}
	
	var savePackage map[string]interface{}
	if err := json.Unmarshal(jsonData, &savePackage); err != nil {
		return fmt.Errorf("failed to unmarshal JSON: %w", err)
	}
	
	data, ok := savePackage["data"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid save format: missing data field")
	}
	
	// Apply migration
	if err := migrator(data); err != nil {
		return fmt.Errorf("migration failed: %w", err)
	}
	
	// Update version
	if metadataMap, ok := savePackage["metadata"].(map[string]interface{}); ok {
		metadataMap["version"] = "current"
	}
	
	// Save migrated file
	jsonData, err = json.MarshalIndent(savePackage, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal migrated data: %w", err)
	}
	
	if err := os.WriteFile(path, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write migrated file: %w", err)
	}
	
	return nil
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
