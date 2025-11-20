package model

import "context"

// Tool represents a tool that can be used by ReAct.
type Tool struct {
	Name        string
	Description string
	Function    func(ctx context.Context, input string) (string, error)
}
