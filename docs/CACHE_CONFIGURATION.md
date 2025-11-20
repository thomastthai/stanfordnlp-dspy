# Cache Configuration for Containerized Environments

This guide explains how to configure DSPy's cache system for containerized environments like Kubernetes and Docker, where ephemeral storage can cause cache loss on container restarts.

## Overview

DSPy caches downloaded datasets to avoid re-downloading on every execution. By default, it uses the system's temporary directory (`/tmp`), which is ephemeral in containerized environments. This document shows how to configure persistent cache storage.

## Environment Variables

DSPy supports the following environment variables for cache configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `DSPY_CACHE_DIR` | Cache directory path | `os.TempDir()` (typically `/tmp`) |
| `DSPY_CACHE_SIZE_MB` | Maximum cache size in megabytes | `1024` (1GB) |
| `DSPY_CACHE_TTL` | Cache time-to-live duration | `24h` |
| `DSPY_CACHE_ENABLED` | Enable/disable caching | `true` |

## Programmatic Configuration

You can also configure the cache programmatically in your Go code:

```go
import (
    "github.com/stanfordnlp/dspy/pkg/dspy"
    "time"
)

func main() {
    // Configure cache directory
    dspy.Configure(
        dspy.WithCacheDir("/persistent/cache"),
        dspy.WithCacheSize(2048),        // 2GB
        dspy.WithCacheTTL(48 * time.Hour),
    )
    
    // Or disable caching entirely
    dspy.Configure(
        dspy.DisableCache(),
    )
}
```

## Kubernetes Deployment

### Using PersistentVolumeClaim

For production Kubernetes deployments, use a PersistentVolumeClaim to ensure cache persistence across pod restarts:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dspy-cache
  namespace: default
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard  # Use your cluster's storage class
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dspy-app
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: dspy
  template:
    metadata:
      labels:
        app: dspy
    spec:
      containers:
      - name: app
        image: your-dspy-app:latest
        env:
        - name: DSPY_CACHE_DIR
          value: "/cache"
        - name: DSPY_CACHE_SIZE_MB
          value: "10240"  # 10GB
        - name: DSPY_CACHE_TTL
          value: "72h"
        volumeMounts:
        - name: cache
          mountPath: /cache
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: dspy-cache
```

### Sharing Cache Across Multiple Pods

To share cache across multiple pods, use `ReadWriteMany` access mode:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dspy-shared-cache
spec:
  accessModes:
    - ReadWriteMany  # Allows multiple pods to mount the same volume
  resources:
    requests:
      storage: 20Gi
  storageClassName: nfs  # Requires a storage class that supports RWX
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dspy-workers
spec:
  replicas: 3  # Multiple pods sharing the same cache
  template:
    spec:
      containers:
      - name: worker
        image: your-dspy-app:latest
        env:
        - name: DSPY_CACHE_DIR
          value: "/shared-cache"
        volumeMounts:
        - name: shared-cache
          mountPath: /shared-cache
      volumes:
      - name: shared-cache
        persistentVolumeClaim:
          claimName: dspy-shared-cache
```

### Using ConfigMap for Cache Configuration

Store cache configuration in a ConfigMap for easier management:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dspy-config
data:
  DSPY_CACHE_DIR: "/cache"
  DSPY_CACHE_SIZE_MB: "5120"
  DSPY_CACHE_TTL: "48h"
  DSPY_CACHE_ENABLED: "true"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dspy-app
spec:
  template:
    spec:
      containers:
      - name: app
        image: your-dspy-app:latest
        envFrom:
        - configMapRef:
            name: dspy-config
        volumeMounts:
        - name: cache
          mountPath: /cache
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: dspy-cache
```

## Docker Deployment

### Using Named Volumes

```bash
# Create a named volume for persistent cache
docker volume create dspy-cache

# Run container with volume mount and environment variables
docker run -d \
  --name dspy-app \
  -e DSPY_CACHE_DIR=/cache \
  -e DSPY_CACHE_SIZE_MB=5120 \
  -e DSPY_CACHE_TTL=48h \
  -v dspy-cache:/cache \
  your-dspy-app:latest
```

### Using Bind Mounts

```bash
# Create local cache directory
mkdir -p /path/to/local/cache

# Run container with bind mount
docker run -d \
  --name dspy-app \
  -e DSPY_CACHE_DIR=/cache \
  -v /path/to/local/cache:/cache \
  your-dspy-app:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  dspy-app:
    image: your-dspy-app:latest
    environment:
      - DSPY_CACHE_DIR=/cache
      - DSPY_CACHE_SIZE_MB=5120
      - DSPY_CACHE_TTL=48h
    volumes:
      - dspy-cache:/cache
    restart: unless-stopped

volumes:
  dspy-cache:
    driver: local
```

## Dockerfile Best Practices

### Single-Stage Build

```dockerfile
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o dspy-app ./cmd/your-app

FROM alpine:latest

# Create cache directory
RUN mkdir -p /cache && chmod 755 /cache

# Copy binary
COPY --from=builder /app/dspy-app /usr/local/bin/

# Set default environment variables
ENV DSPY_CACHE_DIR=/cache \
    DSPY_CACHE_SIZE_MB=1024 \
    DSPY_CACHE_TTL=24h \
    DSPY_CACHE_ENABLED=true

# Use non-root user
RUN addgroup -g 1000 dspy && \
    adduser -D -u 1000 -G dspy dspy && \
    chown -R dspy:dspy /cache

USER dspy

ENTRYPOINT ["dspy-app"]
```

## Cloud Provider Examples

### AWS EKS with EBS

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dspy-cache
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: gp3  # AWS EBS GP3 storage class
  resources:
    requests:
      storage: 20Gi
```

### GCP GKE with Persistent Disk

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dspy-cache
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: standard-rwo  # GCP standard persistent disk
  resources:
    requests:
      storage: 20Gi
```

### Azure AKS with Azure Disk

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dspy-cache
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: managed-premium  # Azure Premium SSD
  resources:
    requests:
      storage: 20Gi
```

## Cache Management

### Monitoring Cache Size

You can monitor cache usage in your application:

```go
package main

import (
    "fmt"
    "os"
    "path/filepath"
)

func getCacheSize(cacheDir string) (int64, error) {
    var size int64
    err := filepath.Walk(cacheDir, func(_ string, info os.FileInfo, err error) error {
        if err != nil {
            return err
        }
        if !info.IsDir() {
            size += info.Size()
        }
        return nil
    })
    return size, err
}

func main() {
    cacheDir := os.Getenv("DSPY_CACHE_DIR")
    if cacheDir == "" {
        cacheDir = os.TempDir()
    }
    
    size, err := getCacheSize(cacheDir)
    if err != nil {
        fmt.Printf("Error getting cache size: %v\n", err)
        return
    }
    
    fmt.Printf("Cache size: %.2f MB\n", float64(size)/(1024*1024))
}
```

### Clearing Cache

To clear the cache in a running container:

```bash
# Via kubectl
kubectl exec -it <pod-name> -- rm -rf /cache/*

# Via docker
docker exec <container-name> rm -rf /cache/*
```

### Pre-populating Cache

To pre-populate cache in a Kubernetes Job before deploying the main application:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: dspy-cache-warmup
spec:
  template:
    spec:
      containers:
      - name: warmup
        image: your-dspy-app:latest
        command: ["/usr/local/bin/cache-warmup.sh"]
        env:
        - name: DSPY_CACHE_DIR
          value: "/cache"
        volumeMounts:
        - name: cache
          mountPath: /cache
      restartPolicy: OnFailure
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: dspy-cache
```

## Troubleshooting

### Issue: Cache Not Persisting

**Symptoms:** Datasets are re-downloaded on every pod restart.

**Solution:** Ensure your PVC is properly mounted and the cache directory is writable:

```bash
# Check if PVC is bound
kubectl get pvc dspy-cache

# Check if volume is mounted in pod
kubectl describe pod <pod-name>

# Check directory permissions
kubectl exec -it <pod-name> -- ls -la /cache
```

### Issue: Permission Denied

**Symptoms:** Cannot write to cache directory.

**Solution:** Ensure the container user has write permissions:

```dockerfile
# In your Dockerfile
RUN chown -R 1000:1000 /cache
USER 1000
```

Or use an init container:

```yaml
initContainers:
- name: fix-permissions
  image: busybox
  command: ['sh', '-c', 'chown -R 1000:1000 /cache']
  volumeMounts:
  - name: cache
    mountPath: /cache
```

### Issue: Out of Disk Space

**Symptoms:** Cache writes fail or pod is evicted.

**Solution:** 
1. Increase PVC size
2. Reduce `DSPY_CACHE_SIZE_MB`
3. Set shorter `DSPY_CACHE_TTL`

```bash
# Increase PVC size (if storage class supports expansion)
kubectl patch pvc dspy-cache -p '{"spec":{"resources":{"requests":{"storage":"50Gi"}}}}'
```

## Performance Considerations

1. **Storage Class Selection:**
   - Use SSD-backed storage for better I/O performance
   - Consider throughput requirements for your workload

2. **Cache Size:**
   - Start with 5-10GB for typical datasets
   - Monitor usage and adjust as needed

3. **Cache TTL:**
   - Balance between freshness and network usage
   - Longer TTL for stable datasets
   - Shorter TTL for frequently updated data

4. **Network Considerations:**
   - For shared cache (RWX), consider network latency
   - Use local volumes when possible for single-pod deployments

## Security Best Practices

1. **Run as non-root user:**
   ```yaml
   securityContext:
     runAsUser: 1000
     runAsGroup: 1000
     fsGroup: 1000
   ```

2. **Read-only root filesystem:**
   ```yaml
   securityContext:
     readOnlyRootFilesystem: true
   ```

3. **Resource limits:**
   ```yaml
   resources:
     limits:
       memory: "2Gi"
       cpu: "1000m"
       ephemeral-storage: "5Gi"
   ```

## Examples

See the `/examples` directory for complete working examples of DSPy applications using configurable cache in various deployment scenarios.
