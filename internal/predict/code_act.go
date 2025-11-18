// Package predict provides prediction modules that interact with language models.
package predict

import (
	"context"
	"fmt"

	"github.com/stanfordnlp/dspy/internal/primitives"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

// CodeAct combines ReAct with code execution for problem solving.
type CodeAct struct {
	*primitives.BaseModule

	// Signature defines the input and output structure
	Signature *signatures.Signature

	// Tools available to the agent
	Tools map[string]*Tool

	// MaxIters is the maximum number of iterations
	MaxIters int

	// CodeActModule generates code for each iteration
	CodeActModule *Predict

	// ExtractModule extracts final answer from trajectory
	ExtractModule *ChainOfThought

	// Interpreter executes Python code (optional)
	Interpreter interface{}

	// Config contains additional configuration
	Config map[string]interface{}
}

// NewCodeAct creates a new CodeAct module.
func NewCodeAct(sig interface{}, tools []*Tool, maxIters int) (*CodeAct, error) {
	if maxIters <= 0 {
		maxIters = 5 // Default to 5 iterations
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

	// Convert tools slice to map
	toolsMap := make(map[string]*Tool)
	for _, tool := range tools {
		toolsMap[tool.Name] = tool
	}

	// Create temporary CodeAct instance for helper methods
	codeAct := &CodeAct{
		BaseModule: primitives.NewBaseModule(),
		Signature:  signature,
		Tools:      toolsMap,
		MaxIters:   maxIters,
		Config:     make(map[string]interface{}),
	}

	// Build CodeAct signature
	codeActSig := codeAct.buildCodeActSignature(signature, toolsMap)

	// Build extraction signature
	extractSig := codeAct.buildExtractionSignature(signature)

	// Create internal modules
	codeActModule, err := New(codeActSig)
	if err != nil {
		return nil, fmt.Errorf("failed to create codeact module: %w", err)
	}

	extractModule, err := NewChainOfThought(extractSig)
	if err != nil {
		return nil, fmt.Errorf("failed to create extract module: %w", err)
	}

	codeAct.CodeActModule = codeActModule
	codeAct.ExtractModule = extractModule

	return codeAct, nil
}

// buildCodeActSignature creates the signature for code generation.
func (c *CodeAct) buildCodeActSignature(baseSig *signatures.Signature, tools map[string]*Tool) *signatures.Signature {
	sig := &signatures.Signature{
		InputFields:  make([]*signatures.Field, len(baseSig.InputFields)),
		OutputFields: make([]*signatures.Field, 0),
		Instructions: c.buildCodeActInstructions(baseSig, tools),
	}

	copy(sig.InputFields, baseSig.InputFields)

	// Add trajectory input field
	trajectoryField := signatures.NewInputField("trajectory")
	trajectoryField.Description = "Past trajectory of code and observations"
	sig.InputFields = append(sig.InputFields, trajectoryField)

	// Add generated_code output field
	codeField := signatures.NewOutputField("generated_code")
	codeField.Description = "Python code that when executed, produces output relevant to answering the question"
	sig.OutputFields = append(sig.OutputFields, codeField)

	// Add finished output field
	finishedField := signatures.NewOutputField("finished")
	finishedField.Description = "Boolean flag to determine if the process is done"
	sig.OutputFields = append(sig.OutputFields, finishedField)

	return sig
}

// buildExtractionSignature creates the signature for extracting final answers.
func (c *CodeAct) buildExtractionSignature(baseSig *signatures.Signature) *signatures.Signature {
	sig := &signatures.Signature{
		InputFields:  make([]*signatures.Field, 0),
		OutputFields: make([]*signatures.Field, len(baseSig.OutputFields)),
		Instructions: baseSig.Instructions,
	}

	// Include original inputs
	sig.InputFields = append(sig.InputFields, baseSig.InputFields...)

	// Add trajectory as input
	trajectoryField := signatures.NewInputField("trajectory")
	trajectoryField.Description = "Complete trajectory of code and observations"
	sig.InputFields = append(sig.InputFields, trajectoryField)

	// Include original outputs
	copy(sig.OutputFields, baseSig.OutputFields)

	return sig
}

// buildCodeActInstructions builds the instruction string.
func (c *CodeAct) buildCodeActInstructions(baseSig *signatures.Signature, tools map[string]*Tool) string {
	instructions := ""
	if baseSig.Instructions != "" {
		instructions = baseSig.Instructions + "\n\n"
	}

	instructions += "You are an intelligent agent. Generate executable Python code to solve the task.\n"
	instructions += "The code should be enclosed in a fenced code block.\n"
	instructions += "When all information is available, mark finished=true.\n"

	if len(tools) > 0 {
		instructions += "\nAvailable functions:\n"
		i := 1
		for _, tool := range tools {
			instructions += fmt.Sprintf("%d. %s: %s\n", i, tool.Name, tool.Description)
			i++
		}
	}

	return instructions
}

// Forward executes the CodeAct loop.
func (c *CodeAct) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	trajectory := make(map[string]interface{})
	maxIters := c.MaxIters

	// Allow overriding max iterations
	if maxItersInterface, ok := inputs["max_iters"]; ok {
		if maxItersInt, ok := maxItersInterface.(int); ok {
			maxIters = maxItersInt
			delete(inputs, "max_iters")
		}
	}

	finished := false

	for idx := 0; idx < maxIters && !finished; idx++ {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Add trajectory to inputs
		codeActInputs := c.copyInputs(inputs)
		codeActInputs["trajectory"] = c.formatTrajectory(trajectory)

		// Generate code
		pred, err := c.CodeActModule.Forward(ctx, codeActInputs)
		if err != nil {
			return nil, fmt.Errorf("codeact step %d failed: %w", idx, err)
		}

		outputs := pred.Fields()
		code := fmt.Sprintf("%v", outputs["generated_code"])
		finishedInterface := outputs["finished"]

		// Check if finished
		if finishedBool, ok := finishedInterface.(bool); ok {
			finished = finishedBool
		} else if finishedStr, ok := finishedInterface.(string); ok {
			finished = (finishedStr == "true" || finishedStr == "True")
		}

		// Parse and execute code
		parsedCode, parseErr := c.parseCode(code)

		if parseErr != "" {
			trajectory[fmt.Sprintf("observation_%d", idx)] = fmt.Sprintf("Failed to parse code: %s", parseErr)
			continue
		}

		trajectory[fmt.Sprintf("generated_code_%d", idx)] = parsedCode

		output, execErr := c.executeCode(parsedCode)
		if execErr != "" {
			trajectory[fmt.Sprintf("observation_%d", idx)] = fmt.Sprintf("Failed to execute code: %s", execErr)
		} else {
			trajectory[fmt.Sprintf("code_output_%d", idx)] = output
		}

		if finished {
			break
		}
	}

	// Extract final answer
	extractInputs := c.copyInputs(inputs)
	extractInputs["trajectory"] = c.formatTrajectory(trajectory)

	extract, err := c.ExtractModule.Forward(ctx, extractInputs)
	if err != nil {
		return nil, fmt.Errorf("extraction failed: %w", err)
	}

	// Add trajectory to final prediction
	finalOutputs := extract.Fields()
	finalOutputs["trajectory"] = trajectory

	result := primitives.NewPrediction(finalOutputs)
	result.SetMetadata("codeact_iterations", len(trajectory))

	return result, nil
}

// parseCode extracts Python code from the generated output.
func (c *CodeAct) parseCode(code string) (string, string) {
	// Reuse the same parsing logic as ProgramOfThought
	pot := &ProgramOfThought{}
	outputs := map[string]interface{}{"generated_code": code}
	return pot.parseCode(outputs)
}

// executeCode simulates code execution.
func (c *CodeAct) executeCode(code string) (string, string) {
	// TODO: Implement actual Python code execution
	// For now, return a simulated success
	if c.Interpreter != nil {
		// Future: Call actual interpreter
	}

	// Simulate execution
	return fmt.Sprintf("[Code execution output: %s]", code), ""
}

// formatTrajectory converts the trajectory map to a human-readable string.
func (c *CodeAct) formatTrajectory(trajectory map[string]interface{}) string {
	if len(trajectory) == 0 {
		return "No previous actions."
	}

	formatted := ""
	i := 0
	for {
		codeKey := fmt.Sprintf("generated_code_%d", i)
		if _, ok := trajectory[codeKey]; !ok {
			break
		}

		formatted += fmt.Sprintf("\nIteration %d:\n", i+1)
		formatted += fmt.Sprintf("Code: %v\n", trajectory[codeKey])

		if output, ok := trajectory[fmt.Sprintf("code_output_%d", i)]; ok {
			formatted += fmt.Sprintf("Output: %v\n", output)
		}

		if obs, ok := trajectory[fmt.Sprintf("observation_%d", i)]; ok {
			formatted += fmt.Sprintf("Observation: %v\n", obs)
		}

		i++
	}

	return formatted
}

// copyInputs creates a copy of the inputs map.
func (c *CodeAct) copyInputs(inputs map[string]interface{}) map[string]interface{} {
	copied := make(map[string]interface{})
	for k, v := range inputs {
		copied[k] = v
	}
	return copied
}

// Copy creates a deep copy of the CodeAct module.
func (c *CodeAct) Copy() primitives.Module {
	newCodeAct := &CodeAct{
		BaseModule:    primitives.NewBaseModule(),
		Signature:     c.Signature,
		Tools:         c.Tools, // Tools are functions, safe to share
		MaxIters:      c.MaxIters,
		CodeActModule: c.CodeActModule.Copy().(*Predict),
		ExtractModule: c.ExtractModule.Copy().(*ChainOfThought),
		Interpreter:   c.Interpreter,
		Config:        make(map[string]interface{}),
	}

	for k, v := range c.Config {
		newCodeAct.Config[k] = v
	}

	return newCodeAct
}

// NamedParameters returns all parameters in this module.
func (c *CodeAct) NamedParameters() []primitives.NamedParameter {
	params := c.CodeActModule.NamedParameters()
	params = append(params, c.ExtractModule.NamedParameters()...)
	return params
}
