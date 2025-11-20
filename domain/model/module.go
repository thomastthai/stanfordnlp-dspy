package model

import "context"

// Module is the core interface for prediction modules.
type Module interface {
	Forward(ctx context.Context, inputs map[string]Field) (*Prediction, error)
	Name() string
}
