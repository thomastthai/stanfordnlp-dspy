# Cache Configuration Example

This example demonstrates how to configure DSPy's cache system for different deployment scenarios.

## Running the Example

### Local Development

```bash
# Use default cache (system temp directory)
go run main.go

# Use custom cache directory
export DSPY_CACHE_DIR=/path/to/cache
go run main.go

# With custom size and TTL
export DSPY_CACHE_DIR=/path/to/cache
export DSPY_CACHE_SIZE_MB=5120
export DSPY_CACHE_TTL=72h
go run main.go
```

### Docker

Build and run with Docker:

```bash
# Build the image
docker build -t dspy-cache-example .

# Run with named volume
docker volume create dspy-cache
docker run --rm \
  -e DSPY_CACHE_DIR=/cache \
  -e DSPY_CACHE_SIZE_MB=2048 \
  -v dspy-cache:/cache \
  dspy-cache-example

# Run with bind mount
mkdir -p /tmp/dspy-local-cache
docker run --rm \
  -e DSPY_CACHE_DIR=/cache \
  -v /tmp/dspy-local-cache:/cache \
  dspy-cache-example
```

### Kubernetes

Deploy to Kubernetes with persistent cache:

```bash
# Apply the manifests
kubectl apply -f k8s/

# Check the pod
kubectl get pods -l app=dspy-cache-example

# View logs
kubectl logs -f deployment/dspy-cache-example

# Check cache volume
kubectl get pvc dspy-cache

# Exec into pod to inspect cache
kubectl exec -it deployment/dspy-cache-example -- ls -la /cache
```

## What This Example Demonstrates

1. **Environment Variable Configuration**: Reading cache settings from environment variables
2. **Programmatic Configuration**: Setting cache options in code
3. **Dataset Loading**: How datasets use the configured cache
4. **Cache Management**: Enabling/disabling cache
5. **Best Practices**: Tips for different deployment scenarios

## Key Concepts

### Cache Directory Precedence

1. Explicit `CacheDir` in dataset config (highest priority)
2. `DSPY_CACHE_DIR` environment variable
3. Default `os.TempDir()` (lowest priority)

### Environment Variables

- `DSPY_CACHE_DIR`: Base cache directory
- `DSPY_CACHE_SIZE_MB`: Maximum cache size in megabytes
- `DSPY_CACHE_TTL`: Cache time-to-live (e.g., "24h", "48h")
- `DSPY_CACHE_ENABLED`: Enable/disable caching ("true" or "false")

### Programmatic Options

- `dspy.WithCacheDir(dir)`: Set cache directory
- `dspy.WithCacheSize(sizeMB)`: Set max cache size
- `dspy.WithCacheTTL(duration)`: Set cache TTL
- `dspy.DisableCache()`: Disable caching entirely

## Related Documentation

See [docs/CACHE_CONFIGURATION.md](../../docs/CACHE_CONFIGURATION.md) for comprehensive documentation on cache configuration for production deployments.
