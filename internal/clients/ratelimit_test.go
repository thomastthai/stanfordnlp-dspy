package clients

import (
	"context"
	"testing"
	"time"
)

func TestNewRateLimiter(t *testing.T) {
	rl := NewRateLimiter(60)
	defer rl.Stop()

	if rl.RequestsPerMinute() != 60 {
		t.Errorf("got %d requests per minute, want 60", rl.RequestsPerMinute())
	}
}

func TestRateLimiterWait(t *testing.T) {
	// Create a rate limiter with 10 requests per minute
	rl := NewRateLimiter(10)
	defer rl.Stop()

	ctx := context.Background()

	// First request should not block
	start := time.Now()
	err := rl.Wait(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	elapsed := time.Since(start)

	if elapsed > 100*time.Millisecond {
		t.Errorf("first request took too long: %v", elapsed)
	}
}

func TestRateLimiterContext(t *testing.T) {
	// Create a rate limiter with very low rate
	rl := NewRateLimiter(1)
	defer rl.Stop()

	// Consume the available token
	ctx := context.Background()
	err := rl.Wait(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Create a context with short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// This should timeout
	err = rl.Wait(ctx)
	if err != context.DeadlineExceeded {
		t.Errorf("expected DeadlineExceeded, got %v", err)
	}
}

func TestRateLimiterSetRate(t *testing.T) {
	rl := NewRateLimiter(10)
	defer rl.Stop()

	// Change the rate
	rl.SetRate(20)

	if rl.RequestsPerMinute() != 20 {
		t.Errorf("got %d requests per minute, want 20", rl.RequestsPerMinute())
	}
}

func TestRateLimiterNoLimit(t *testing.T) {
	// Create a rate limiter with no limit (0 or negative)
	rl := NewRateLimiter(0)
	defer rl.Stop()

	ctx := context.Background()

	// Should not block at all
	for i := 0; i < 100; i++ {
		err := rl.Wait(ctx)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
}

func TestRateLimiterStop(t *testing.T) {
	rl := NewRateLimiter(60)

	// Stop should not panic
	rl.Stop()

	// Calling stop again should not panic
	rl.Stop()
}

func TestRateLimiterConcurrent(t *testing.T) {
	rl := NewRateLimiter(100)
	defer rl.Stop()

	ctx := context.Background()
	errCh := make(chan error, 10)

	// Launch multiple goroutines
	for i := 0; i < 10; i++ {
		go func() {
			err := rl.Wait(ctx)
			errCh <- err
		}()
	}

	// Check that all completed without error
	for i := 0; i < 10; i++ {
		select {
		case err := <-errCh:
			if err != nil {
				t.Errorf("goroutine %d got error: %v", i, err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("timeout waiting for goroutines")
		}
	}
}
