package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/stanfordnlp/dspy/pkg/dspy"
	"github.com/stanfordnlp/dspy/pkg/dspy/datasets"
)

func main() {
	fmt.Println("=== DSPy Cache Configuration Example ===\n")

	// Example 1: Using environment variables (recommended for containers)
	fmt.Println("1. Environment Variable Configuration:")
	if cacheDir := os.Getenv("DSPY_CACHE_DIR"); cacheDir != "" {
		fmt.Printf("   DSPY_CACHE_DIR: %s\n", cacheDir)
	} else {
		fmt.Printf("   DSPY_CACHE_DIR: (not set, using default: %s)\n", os.TempDir())
	}

	// Example 2: Programmatic configuration
	fmt.Println("\n2. Programmatic Configuration:")
	dspy.Configure(
		dspy.WithCacheDir("/tmp/dspy-custom-cache"),
		dspy.WithCacheSize(2048), // 2GB
		dspy.WithCacheTTL(48*time.Hour),
	)

	settings := dspy.GetSettings()
	fmt.Printf("   Cache Directory: %s\n", settings.CacheConfig.Dir)
	fmt.Printf("   Max Cache Size: %d MB\n", settings.CacheConfig.MaxSizeMB)
	fmt.Printf("   Cache TTL: %v\n", settings.CacheConfig.TTL)
	fmt.Printf("   Cache Enabled: %t\n", settings.CacheConfig.Enabled)

	// Example 3: Dataset with custom cache
	fmt.Println("\n3. Loading Dataset with Configured Cache:")
	squad := datasets.NewSQuAD(datasets.SQuADConfig{
		Version: "v2.0",
		// CacheDir is automatically read from environment or settings
	})

	fmt.Printf("   Dataset cache directory: %s/dspy/squad\n", getCacheBaseDir())

	// Example 4: Disabling cache
	fmt.Println("\n4. Disabling Cache:")
	dspy.Configure(
		dspy.DisableCache(),
	)
	settings = dspy.GetSettings()
	fmt.Printf("   Cache Enabled: %t\n", settings.CacheConfig.Enabled)

	// Re-enable for demonstration
	dspy.Configure(
		dspy.WithCache(true),
	)

	// Example 5: Loading a small dataset to demonstrate caching
	fmt.Println("\n5. Testing Dataset Load (this may download data):")
	fmt.Println("   Loading SQuAD dataset...")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Note: This will attempt to download if not cached
	// In a real scenario with network access, this would work
	_, err := squad.Load(ctx)
	if err != nil {
		fmt.Printf("   Note: Dataset load failed (expected without network): %v\n", err)
		fmt.Println("   In a real deployment with network access, this would work!")
	} else {
		fmt.Printf("   Successfully loaded %d examples\n", squad.Len())
	}

	fmt.Println("\n=== Cache Configuration Tips ===")
	fmt.Println("For Kubernetes:")
	fmt.Println("  - Set DSPY_CACHE_DIR=/cache")
	fmt.Println("  - Mount a PersistentVolumeClaim to /cache")
	fmt.Println("  - Use StorageClass with appropriate performance")
	fmt.Println("\nFor Docker:")
	fmt.Println("  - Use named volumes: docker volume create dspy-cache")
	fmt.Println("  - Mount with: -v dspy-cache:/cache")
	fmt.Println("  - Set environment: -e DSPY_CACHE_DIR=/cache")
	fmt.Println("\nFor local development:")
	fmt.Println("  - Default cache in /tmp works fine")
	fmt.Println("  - Or set DSPY_CACHE_DIR to a local directory")
}

func getCacheBaseDir() string {
	if dir := os.Getenv("DSPY_CACHE_DIR"); dir != "" {
		return dir
	}
	return os.TempDir()
}

// init demonstrates how to configure cache at application startup
func init() {
	// Check if running in a container (common environment variable)
	if os.Getenv("KUBERNETES_SERVICE_HOST") != "" {
		log.Println("Detected Kubernetes environment")
		// In K8s, you would typically use a mounted volume
		// The DSPY_CACHE_DIR env var would be set in the deployment
	}

	if os.Getenv("DOCKER_CONTAINER") != "" {
		log.Println("Detected Docker environment")
		// In Docker, use a mounted volume
	}
}
