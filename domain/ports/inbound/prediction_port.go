package inbound

import (
	"context"
	"github.com/stanfordnlp/dspy/domain/model"
)

// PredictionPort defines prediction operations.
type PredictionPort interface {
	Predict(ctx context.Context, signature string, inputs map[string]interface{}) (*model.Prediction, error)
	ChainOfThought(ctx context.Context, signature string, inputs map[string]interface{}) (*model.Prediction, error)
	ReAct(ctx context.Context, signature string, inputs map[string]interface{}, tools []*model.Tool) (*model.Prediction, error)
}
