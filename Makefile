.PHONY: all build test lint clean install deps fmt vet help

# Variables
GO := go
GOFLAGS := -v
BINARY_NAME := dspy
CMD_DIR := ./cmd/dspy
BUILD_DIR := ./build
PKG := ./...

# Default target
all: deps fmt vet test build

## help: Show this help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^##' $(MAKEFILE_LIST) | sed 's/^## /  /'

## deps: Download and install dependencies
deps:
	@echo "==> Installing dependencies..."
	$(GO) mod download
	$(GO) mod tidy

## build: Build the binary
build: deps
	@echo "==> Building $(BINARY_NAME)..."
	@mkdir -p $(BUILD_DIR)
	$(GO) build $(GOFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME) $(CMD_DIR)

## install: Install the binary
install: deps
	@echo "==> Installing $(BINARY_NAME)..."
	$(GO) install $(GOFLAGS) $(CMD_DIR)

## test: Run tests
test:
	@echo "==> Running tests..."
	$(GO) test $(GOFLAGS) -race -coverprofile=coverage.out $(PKG)

## test-coverage: Run tests with coverage report
test-coverage: test
	@echo "==> Generating coverage report..."
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

## bench: Run benchmarks
bench:
	@echo "==> Running benchmarks..."
	$(GO) test -bench=. -benchmem $(PKG)

## fmt: Format code
fmt:
	@echo "==> Formatting code..."
	$(GO) fmt $(PKG)

## vet: Run go vet
vet:
	@echo "==> Running go vet..."
	$(GO) vet $(PKG)

## lint: Run linters (requires golangci-lint)
lint:
	@echo "==> Running linters..."
	@which golangci-lint > /dev/null 2>&1 || (echo "golangci-lint not found. Install it from https://golangci-lint.run/usage/install/" && exit 1)
	golangci-lint run $(PKG)

## clean: Remove build artifacts
clean:
	@echo "==> Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -f coverage.out coverage.html
	$(GO) clean

## run: Run the application
run: build
	@echo "==> Running $(BINARY_NAME)..."
	$(BUILD_DIR)/$(BINARY_NAME)

## mod-update: Update all dependencies
mod-update:
	@echo "==> Updating dependencies..."
	$(GO) get -u $(PKG)
	$(GO) mod tidy

## check: Run all checks (fmt, vet, test)
check: fmt vet test

.DEFAULT_GOAL := help
