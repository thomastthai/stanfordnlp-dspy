package monitoring

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// PrometheusMonitor implements Monitor using Prometheus metrics.
type PrometheusMonitor struct {
	requestDuration *prometheus.HistogramVec
	requestCount    *prometheus.CounterVec
	errorCount      *prometheus.CounterVec
	tokenCount      *prometheus.CounterVec
	costCounter     *prometheus.CounterVec
	cacheHits       *prometheus.CounterVec
	cacheMisses     *prometheus.CounterVec
}

// NewPrometheusMonitor creates a new Prometheus monitor.
func NewPrometheusMonitor(namespace string) *PrometheusMonitor {
	if namespace == "" {
		namespace = "dspy"
	}

	return &PrometheusMonitor{
		requestDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: namespace,
				Name:      "request_duration_seconds",
				Help:      "Duration of requests in seconds",
				Buckets:   prometheus.ExponentialBuckets(0.001, 2, 15), // 1ms to ~16s
			},
			[]string{"model"},
		),
		requestCount: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "requests_total",
				Help:      "Total number of requests",
			},
			[]string{"model"},
		),
		errorCount: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "errors_total",
				Help:      "Total number of errors",
			},
			[]string{"model", "error_type"},
		),
		tokenCount: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "tokens_total",
				Help:      "Total number of tokens processed",
			},
			[]string{"model"},
		),
		costCounter: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "cost_total",
				Help:      "Total cost in dollars",
			},
			[]string{"model"},
		),
		cacheHits: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "cache_hits_total",
				Help:      "Total number of cache hits",
			},
			[]string{"cache_type"},
		),
		cacheMisses: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "cache_misses_total",
				Help:      "Total number of cache misses",
			},
			[]string{"cache_type"},
		),
	}
}

// RecordRequest implements Monitor.RecordRequest.
func (m *PrometheusMonitor) RecordRequest(duration time.Duration, model string, tokens int) {
	m.requestDuration.WithLabelValues(model).Observe(duration.Seconds())
	m.requestCount.WithLabelValues(model).Inc()
	if tokens > 0 {
		m.tokenCount.WithLabelValues(model).Add(float64(tokens))
	}
}

// RecordError implements Monitor.RecordError.
func (m *PrometheusMonitor) RecordError(err error, model string) {
	errorType := "unknown"
	if err != nil {
		errorType = err.Error()
		// Truncate long error messages
		if len(errorType) > 50 {
			errorType = errorType[:50]
		}
	}
	m.errorCount.WithLabelValues(model, errorType).Inc()
}

// RecordCost implements Monitor.RecordCost.
func (m *PrometheusMonitor) RecordCost(cost float64, model string) {
	m.costCounter.WithLabelValues(model).Add(cost)
}

// RecordCacheHit implements Monitor.RecordCacheHit.
func (m *PrometheusMonitor) RecordCacheHit(cacheType string) {
	m.cacheHits.WithLabelValues(cacheType).Inc()
}

// RecordCacheMiss implements Monitor.RecordCacheMiss.
func (m *PrometheusMonitor) RecordCacheMiss(cacheType string) {
	m.cacheMisses.WithLabelValues(cacheType).Inc()
}
