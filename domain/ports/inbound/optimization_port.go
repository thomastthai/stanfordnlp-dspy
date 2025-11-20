package inbound

import (
	"context"
	"github.com/stanfordnlp/dspy/domain/model"
)

// OptimizationPort defines optimization operations.
type OptimizationPort interface {
	Bootstrap(ctx context.Context, module model.Module, trainset []*model.Example) (model.Module, error)
	MIPRO(ctx context.Context, module model.Module, trainset []*model.Example, metric Metric) (model.Module, error)
}

type Metric func(example *model.Example, prediction *model.Prediction) float64
