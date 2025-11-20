package outbound

import (
	"context"
	"github.com/stanfordnlp/dspy/domain/model"
)

// DatasetRepositoryPort defines operations for dataset persistence.
type DatasetRepositoryPort interface {
	Load(ctx context.Context, name string) ([]*model.Example, error)
	Save(ctx context.Context, name string, examples []*model.Example) error
	List(ctx context.Context) ([]string, error)
}
