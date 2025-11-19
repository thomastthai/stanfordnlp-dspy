// Package monitoring provides observability features for DSPy.
package monitoring

import (
	"context"
	"time"
)

// Monitor provides methods for recording metrics and traces.
type Monitor interface {
	// RecordRequest records a completed request
	RecordRequest(duration time.Duration, model string, tokens int)
	
	// RecordError records an error
	RecordError(err error, model string)
	
	// RecordCost records the cost of an operation
	RecordCost(cost float64, model string)
	
	// RecordCacheHit records a cache hit
	RecordCacheHit(cacheType string)
	
	// RecordCacheMiss records a cache miss
	RecordCacheMiss(cacheType string)
}

// Tracer provides distributed tracing capabilities.
type Tracer interface {
	// StartSpan starts a new span
	StartSpan(ctx context.Context, name string) (context.Context, Span)
}

// Span represents a trace span.
type Span interface {
	// End ends the span
	End()
	
	// SetAttribute sets an attribute on the span
	SetAttribute(key string, value interface{})
	
	// SetError marks the span as errored
	SetError(err error)
	
	// Context returns the span context
	Context() context.Context
}

// NoOpMonitor is a no-op implementation of Monitor.
type NoOpMonitor struct{}

// RecordRequest implements Monitor.RecordRequest.
func (m *NoOpMonitor) RecordRequest(duration time.Duration, model string, tokens int) {}

// RecordError implements Monitor.RecordError.
func (m *NoOpMonitor) RecordError(err error, model string) {}

// RecordCost implements Monitor.RecordCost.
func (m *NoOpMonitor) RecordCost(cost float64, model string) {}

// RecordCacheHit implements Monitor.RecordCacheHit.
func (m *NoOpMonitor) RecordCacheHit(cacheType string) {}

// RecordCacheMiss implements Monitor.RecordCacheMiss.
func (m *NoOpMonitor) RecordCacheMiss(cacheType string) {}

// NoOpTracer is a no-op implementation of Tracer.
type NoOpTracer struct{}

// StartSpan implements Tracer.StartSpan.
func (t *NoOpTracer) StartSpan(ctx context.Context, name string) (context.Context, Span) {
	return ctx, &NoOpSpan{ctx: ctx}
}

// NoOpSpan is a no-op implementation of Span.
type NoOpSpan struct {
	ctx context.Context
}

// End implements Span.End.
func (s *NoOpSpan) End() {}

// SetAttribute implements Span.SetAttribute.
func (s *NoOpSpan) SetAttribute(key string, value interface{}) {}

// SetError implements Span.SetError.
func (s *NoOpSpan) SetError(err error) {}

// Context implements Span.Context.
func (s *NoOpSpan) Context() context.Context {
	return s.ctx
}
