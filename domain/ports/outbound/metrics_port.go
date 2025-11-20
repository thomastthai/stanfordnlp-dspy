package outbound

import (
	"context"
	"time"
)

// MetricsPort defines observability operations.
type MetricsPort interface {
	RecordPrediction(ctx context.Context, duration time.Duration, success bool)
	RecordOptimization(ctx context.Context, metric string, value float64)
	RecordCacheHit(ctx context.Context)
	RecordCacheMiss(ctx context.Context)
}
