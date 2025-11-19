// Package predict provides prediction modules that interact with language models.
package predict

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/stanfordnlp/dspy/internal/primitives"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

// ProgramOfThought generates and executes Python code for reasoning.
type ProgramOfThought struct {
	*primitives.BaseModule

	// Signature defines the input and output structure
	Signature *signatures.Signature

	// MaxIters is the maximum number of code generation retries
	MaxIters int

	// CodeGenerate generates initial code
	CodeGenerate *ChainOfThought

	// CodeRegenerate regenerates code after errors
	CodeRegenerate *ChainOfThought

	// GenerateAnswer extracts final answer from code output
	GenerateAnswer *ChainOfThought

	// Interpreter executes Python code (optional, for future implementation)
	Interpreter interface{}

	// Config contains additional configuration
	Config map[string]interface{}
}

// NewProgramOfThought creates a new ProgramOfThought module.
func NewProgramOfThought(sig interface{}, maxIters int) (*ProgramOfThought, error) {
	if maxIters <= 0 {
		maxIters = 3 // Default to 3 retries
	}

	var signature *signatures.Signature
	var err error

	switch s := sig.(type) {
	case string:
		signature, err = signatures.NewSignature(s)
		if err != nil {
			return nil, fmt.Errorf("failed to parse signature: %w", err)
		}
	case *signatures.Signature:
		signature = s
	default:
		return nil, fmt.Errorf("signature must be string or *Signature, got %T", sig)
	}

	pot := &ProgramOfThought{
		BaseModule: primitives.NewBaseModule(),
		Signature:  signature,
		MaxIters:   maxIters,
		Config:     make(map[string]interface{}),
	}

	// Build signatures for code generation
	genSig := pot.buildCodeGenerateSignature(signature)
	regenSig := pot.buildCodeRegenerateSignature(signature)
	answerSig := pot.buildAnswerSignature(signature)

	// Create internal modules
	pot.CodeGenerate, err = NewChainOfThought(genSig)
	if err != nil {
		return nil, fmt.Errorf("failed to create code generate module: %w", err)
	}

	pot.CodeRegenerate, err = NewChainOfThought(regenSig)
	if err != nil {
		return nil, fmt.Errorf("failed to create code regenerate module: %w", err)
	}

	pot.GenerateAnswer, err = NewChainOfThought(answerSig)
	if err != nil {
		return nil, fmt.Errorf("failed to create answer module: %w", err)
	}

	return pot, nil
}

// buildCodeGenerateSignature creates signature for initial code generation.
func (p *ProgramOfThought) buildCodeGenerateSignature(baseSig *signatures.Signature) *signatures.Signature {
	sig := &signatures.Signature{
		InputFields:  make([]*signatures.Field, len(baseSig.InputFields)),
		OutputFields: make([]*signatures.Field, 0),
		Instructions: "Generate executable Python code that solves the problem. Use final_answer() to return results.",
	}

	copy(sig.InputFields, baseSig.InputFields)

	// Add generated_code output field
	codeField := signatures.NewOutputField("generated_code")
	codeField.Description = "Python code that answers the question"
	codeField.Prefix = "Code:"
	sig.OutputFields = append(sig.OutputFields, codeField)

	return sig
}

// buildCodeRegenerateSignature creates signature for code regeneration after errors.
func (p *ProgramOfThought) buildCodeRegenerateSignature(baseSig *signatures.Signature) *signatures.Signature {
	sig := &signatures.Signature{
		InputFields:  make([]*signatures.Field, len(baseSig.InputFields)),
		OutputFields: make([]*signatures.Field, 0),
		Instructions: "The previous code had an error. Generate corrected Python code.",
	}

	copy(sig.InputFields, baseSig.InputFields)

	// Add previous_code and error as inputs
	prevCodeField := signatures.NewInputField("previous_code")
	prevCodeField.Description = "Previously-generated Python code that errored"
	sig.InputFields = append(sig.InputFields, prevCodeField)

	errorField := signatures.NewInputField("error")
	errorField.Description = "Error message from the previous code"
	sig.InputFields = append(sig.InputFields, errorField)

	// Add generated_code output field
	codeField := signatures.NewOutputField("generated_code")
	codeField.Description = "Corrected Python code"
	codeField.Prefix = "Code:"
	sig.OutputFields = append(sig.OutputFields, codeField)

	return sig
}

// buildAnswerSignature creates signature for extracting final answer.
func (p *ProgramOfThought) buildAnswerSignature(baseSig *signatures.Signature) *signatures.Signature {
	sig := &signatures.Signature{
		InputFields:  make([]*signatures.Field, len(baseSig.InputFields)),
		OutputFields: make([]*signatures.Field, len(baseSig.OutputFields)),
		Instructions: "Extract the final answer from the code output.",
	}

	copy(sig.InputFields, baseSig.InputFields)

	// Add code and code_output as inputs
	finalCodeField := signatures.NewInputField("final_generated_code")
	finalCodeField.Description = "Final generated Python code"
	sig.InputFields = append(sig.InputFields, finalCodeField)

	codeOutputField := signatures.NewInputField("code_output")
	codeOutputField.Description = "Output from executing the code"
	sig.InputFields = append(sig.InputFields, codeOutputField)

	copy(sig.OutputFields, baseSig.OutputFields)

	return sig
}

// Forward generates and executes code to solve the problem.
func (p *ProgramOfThought) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Generate initial code
	codeData, err := p.CodeGenerate.Forward(ctx, inputs)
	if err != nil {
		return nil, fmt.Errorf("code generation failed: %w", err)
	}

	code, parseErr := p.parseCode(codeData.Fields())
	output, execErr := "", ""

	if parseErr == "" {
		output, execErr = p.executeCode(code)
	} else {
		execErr = parseErr
	}

	// Retry code generation if there were errors
	hop := 1
	for execErr != "" && hop < p.MaxIters {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Add error information for regeneration
		regenInputs := p.copyInputs(inputs)
		regenInputs["previous_code"] = code
		regenInputs["error"] = execErr

		codeData, err = p.CodeRegenerate.Forward(ctx, regenInputs)
		if err != nil {
			return nil, fmt.Errorf("code regeneration failed: %w", err)
		}

		code, parseErr = p.parseCode(codeData.Fields())
		if parseErr == "" {
			output, execErr = p.executeCode(code)
		} else {
			execErr = parseErr
		}

		hop++
	}

	if execErr != "" {
		return nil, fmt.Errorf("failed to execute code after %d attempts: %s", p.MaxIters, execErr)
	}

	// Extract final answer from code output
	answerInputs := p.copyInputs(inputs)
	answerInputs["final_generated_code"] = code
	answerInputs["code_output"] = output

	answer, err := p.GenerateAnswer.Forward(ctx, answerInputs)
	if err != nil {
		return nil, fmt.Errorf("answer extraction failed: %w", err)
	}

	// Add metadata
	answer.SetMetadata("generated_code", code)
	answer.SetMetadata("code_output", output)
	answer.SetMetadata("code_attempts", hop)

	return answer, nil
}

// parseCode extracts Python code from the generated output.
func (p *ProgramOfThought) parseCode(outputs map[string]interface{}) (string, string) {
	codeInterface, ok := outputs["generated_code"]
	if !ok {
		return "", "Error: No generated_code field in output"
	}

	code := fmt.Sprintf("%v", codeInterface)

	// Remove markdown code blocks
	codeBlockRegex := regexp.MustCompile("```python\\s*([\\s\\S]*?)\\s*```")
	matches := codeBlockRegex.FindStringSubmatch(code)
	if len(matches) > 1 {
		code = matches[1]
	}

	code = strings.TrimSpace(code)

	if code == "" {
		return "", "Error: Empty code after parsing"
	}

	return code, ""
}

// executeCode simulates code execution.
// In a real implementation, this would use a Python interpreter.
func (p *ProgramOfThought) executeCode(code string) (string, string) {
	// TODO: Implement actual Python code execution
	// For now, return a simulated success
	if p.Interpreter != nil {
		// Future: Call actual interpreter
	}

	// Simulate execution
	return fmt.Sprintf("[Code execution output: %s]", code), ""
}

// copyInputs creates a copy of the inputs map.
func (p *ProgramOfThought) copyInputs(inputs map[string]interface{}) map[string]interface{} {
	copied := make(map[string]interface{})
	for k, v := range inputs {
		copied[k] = v
	}
	return copied
}

// Copy creates a deep copy of the ProgramOfThought module.
func (p *ProgramOfThought) Copy() primitives.Module {
	newPOT := &ProgramOfThought{
		BaseModule:     primitives.NewBaseModule(),
		Signature:      p.Signature,
		MaxIters:       p.MaxIters,
		CodeGenerate:   p.CodeGenerate.Copy().(*ChainOfThought),
		CodeRegenerate: p.CodeRegenerate.Copy().(*ChainOfThought),
		GenerateAnswer: p.GenerateAnswer.Copy().(*ChainOfThought),
		Interpreter:    p.Interpreter,
		Config:         make(map[string]interface{}),
	}

	for k, v := range p.Config {
		newPOT.Config[k] = v
	}

	return newPOT
}

// NamedParameters returns all parameters in this module.
func (p *ProgramOfThought) NamedParameters() []primitives.NamedParameter {
	params := p.CodeGenerate.NamedParameters()
	params = append(params, p.CodeRegenerate.NamedParameters()...)
	params = append(params, p.GenerateAnswer.NamedParameters()...)
	return params
}
