package signatures

import (
	"testing"
)

func TestParseSignature(t *testing.T) {
	tests := []struct {
		name          string
		input         string
		wantErr       bool
		wantInputs    int
		wantOutputs   int
		checkFields   func(*testing.T, *Signature)
	}{
		{
			name:        "simple signature",
			input:       "question -> answer",
			wantErr:     false,
			wantInputs:  1,
			wantOutputs: 1,
			checkFields: func(t *testing.T, sig *Signature) {
				if sig.InputFields[0].Name != "question" {
					t.Errorf("expected input 'question', got %s", sig.InputFields[0].Name)
				}
				if sig.OutputFields[0].Name != "answer" {
					t.Errorf("expected output 'answer', got %s", sig.OutputFields[0].Name)
				}
			},
		},
		{
			name:        "multiple inputs",
			input:       "question, context -> answer",
			wantErr:     false,
			wantInputs:  2,
			wantOutputs: 1,
			checkFields: func(t *testing.T, sig *Signature) {
				if sig.InputFields[0].Name != "question" {
					t.Errorf("expected input 'question', got %s", sig.InputFields[0].Name)
				}
				if sig.InputFields[1].Name != "context" {
					t.Errorf("expected input 'context', got %s", sig.InputFields[1].Name)
				}
			},
		},
		{
			name:        "multiple outputs",
			input:       "text -> summary, sentiment",
			wantErr:     false,
			wantInputs:  1,
			wantOutputs: 2,
			checkFields: func(t *testing.T, sig *Signature) {
				if sig.OutputFields[0].Name != "summary" {
					t.Errorf("expected output 'summary', got %s", sig.OutputFields[0].Name)
				}
				if sig.OutputFields[1].Name != "sentiment" {
					t.Errorf("expected output 'sentiment', got %s", sig.OutputFields[1].Name)
				}
			},
		},
		{
			name:    "missing arrow",
			input:   "question answer",
			wantErr: true,
		},
		{
			name:    "empty input",
			input:   " -> answer",
			wantErr: true,
		},
		{
			name:    "empty output",
			input:   "question -> ",
			wantErr: true,
		},
		{
			name:        "whitespace handling",
			input:       "  question  ,  context  ->  answer  ",
			wantErr:     false,
			wantInputs:  2,
			wantOutputs: 1,
			checkFields: func(t *testing.T, sig *Signature) {
				if sig.InputFields[0].Name != "question" {
					t.Errorf("expected trimmed 'question', got %q", sig.InputFields[0].Name)
				}
			},
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sig, err := ParseSignature(tt.input)
			
			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}
			
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			
			if len(sig.InputFields) != tt.wantInputs {
				t.Errorf("expected %d inputs, got %d", tt.wantInputs, len(sig.InputFields))
			}
			
			if len(sig.OutputFields) != tt.wantOutputs {
				t.Errorf("expected %d outputs, got %d", tt.wantOutputs, len(sig.OutputFields))
			}
			
			if tt.checkFields != nil {
				tt.checkFields(t, sig)
			}
		})
	}
}

func TestSignature_String(t *testing.T) {
	sig, err := ParseSignature("question, context -> answer")
	if err != nil {
		t.Fatalf("failed to parse signature: %v", err)
	}
	
	s := sig.String()
	expected := "question, context -> answer"
	if s != expected {
		t.Errorf("expected %q, got %q", expected, s)
	}
}

func TestSignature_Validate(t *testing.T) {
	tests := []struct {
		name    string
		sig     *Signature
		wantErr bool
	}{
		{
			name: "valid signature",
			sig: &Signature{
				InputFields:  []*Field{NewInputField("input")},
				OutputFields: []*Field{NewOutputField("output")},
			},
			wantErr: false,
		},
		{
			name: "no inputs",
			sig: &Signature{
				InputFields:  []*Field{},
				OutputFields: []*Field{NewOutputField("output")},
			},
			wantErr: true,
		},
		{
			name: "no outputs",
			sig: &Signature{
				InputFields:  []*Field{NewInputField("input")},
				OutputFields: []*Field{},
			},
			wantErr: true,
		},
		{
			name: "duplicate field names",
			sig: &Signature{
				InputFields: []*Field{
					NewInputField("field"),
					NewInputField("field"),
				},
				OutputFields: []*Field{NewOutputField("output")},
			},
			wantErr: true,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.sig.Validate()
			if tt.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
