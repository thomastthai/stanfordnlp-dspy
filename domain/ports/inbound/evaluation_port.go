package inbound

import (
	"context"
	"github.com/stanfordnlp/dspy/domain/model"
)

// EvaluationPort defines evaluation operations.
type EvaluationPort interface {
	Evaluate(ctx context.Context, module model.Module, testset []*model.Example, metric Metric) (float64, error)
}
