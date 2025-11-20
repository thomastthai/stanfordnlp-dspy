package model

// Signature defines the input/output structure for a module.
type Signature struct {
	Name         string
	Instructions string
	InputFields  []FieldSpec
	OutputFields []FieldSpec
}

type FieldSpec struct {
	Name        string
	Type        string
	Description string
	Required    bool
	Format      string
}
