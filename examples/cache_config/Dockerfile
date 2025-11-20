# Multi-stage build for smaller final image
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
WORKDIR /app/examples/cache_config
RUN CGO_ENABLED=0 GOOS=linux go build -o /cache-config-example .

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates

# Create non-root user
RUN addgroup -g 1000 dspy && \
    adduser -D -u 1000 -G dspy dspy

# Create cache directory
RUN mkdir -p /cache && chown dspy:dspy /cache

# Copy binary from builder
COPY --from=builder /cache-config-example /usr/local/bin/cache-config-example

# Set default environment variables
ENV DSPY_CACHE_DIR=/cache \
    DSPY_CACHE_SIZE_MB=1024 \
    DSPY_CACHE_TTL=24h \
    DSPY_CACHE_ENABLED=true

# Switch to non-root user
USER dspy

# Set working directory
WORKDIR /home/dspy

# Run the application
ENTRYPOINT ["/usr/local/bin/cache-config-example"]
