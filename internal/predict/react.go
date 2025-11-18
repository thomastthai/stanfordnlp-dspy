// Package predict provides prediction modules that interact with language models.
package predict

import (
	"context"
	"fmt"
	"strings"

	"github.com/stanfordnlp/dspy/internal/primitives"
	"github.com/stanfordnlp/dspy/internal/signatures"
)

// Tool represents a callable tool for the ReAct agent.
type Tool struct {
	Name        string
	Description string
	Func        func(map[string]interface{}) (string, error)
}

// ReAct implements the Reasoning and Acting paradigm for tool-using agents.
type ReAct struct {
	*primitives.BaseModule

	// Signature defines the input and output structure
	Signature *signatures.Signature

	// Tools available to the agent
	Tools map[string]*Tool

	// MaxIters is the maximum number of reasoning-acting iterations
	MaxIters int

	// ReactModule is the internal predictor for thought/action
	ReactModule *Predict

	// ExtractModule is the internal predictor for final answer extraction
	ExtractModule *ChainOfThought

	// Config contains additional configuration
	Config map[string]interface{}
}

// NewReAct creates a new ReAct agent module.
func NewReAct(sig interface{}, tools []*Tool, maxIters int) (*ReAct, error) {
	if maxIters <= 0 {
		maxIters = 10 // Default to 10 iterations
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

	// Add "finish" tool
	toolsMap["finish"] = &Tool{
		Name:        "finish",
		Description: "Marks the task as complete and signals that all information is available",
		Func: func(args map[string]interface{}) (string, error) {
			return "Completed.", nil
		},
	}

	// Create a temporary ReAct instance to call helper methods
	react := &ReAct{
		BaseModule: primitives.NewBaseModule(),
		Signature:  signature,
		Tools:      toolsMap,
		MaxIters:   maxIters,
		Config:     make(map[string]interface{}),
	}

	// Build ReAct signature with thought/action/args fields
	reactSig := react.buildReActSignature(signature, toolsMap)

	// Build extraction signature
	extractSig := react.buildExtractionSignature(signature)

	// Create internal modules
	reactModule, err := New(reactSig)
	if err != nil {
		return nil, fmt.Errorf("failed to create react module: %w", err)
	}

	extractModule, err := NewChainOfThought(extractSig)
	if err != nil {
		return nil, fmt.Errorf("failed to create extract module: %w", err)
	}

	react.ReactModule = reactModule
	react.ExtractModule = extractModule

	return react, nil
}

// buildReActSignature creates the signature for the ReAct reasoning loop.
func (r *ReAct) buildReActSignature(baseSig *signatures.Signature, tools map[string]*Tool) *signatures.Signature {
	// Start with input fields from base signature
	reactSig := &signatures.Signature{
		InputFields:  make([]*signatures.Field, len(baseSig.InputFields)),
		OutputFields: make([]*signatures.Field, 0),
		Instructions: r.buildReActInstructions(baseSig, tools),
	}

	copy(reactSig.InputFields, baseSig.InputFields)

	// Add trajectory input field
	trajectoryField := signatures.NewInputField("trajectory")
	trajectoryField.Description = "Past trajectory of thoughts, actions, and observations"
	reactSig.InputFields = append(reactSig.InputFields, trajectoryField)

	// Add output fields for thought, tool name, and tool args
	thoughtField := signatures.NewOutputField("next_thought")
	thoughtField.Description = "Reasoning about the current situation"
	reactSig.OutputFields = append(reactSig.OutputFields, thoughtField)

	toolNameField := signatures.NewOutputField("next_tool_name")
	toolNameField.Description = "Name of the tool to use next"
	reactSig.OutputFields = append(reactSig.OutputFields, toolNameField)

	toolArgsField := signatures.NewOutputField("next_tool_args")
	toolArgsField.Description = "Arguments for the tool in JSON format"
	reactSig.OutputFields = append(reactSig.OutputFields, toolArgsField)

	return reactSig
}

// buildExtractionSignature creates the signature for extracting final answers.
func (r *ReAct) buildExtractionSignature(baseSig *signatures.Signature) *signatures.Signature {
	extractSig := &signatures.Signature{
		InputFields:  make([]*signatures.Field, 0),
		OutputFields: make([]*signatures.Field, len(baseSig.OutputFields)),
		Instructions: baseSig.Instructions,
	}

	// Include original inputs
	extractSig.InputFields = append(extractSig.InputFields, baseSig.InputFields...)

	// Add trajectory as input
	trajectoryField := signatures.NewInputField("trajectory")
	trajectoryField.Description = "Complete trajectory of thoughts, actions, and observations"
	extractSig.InputFields = append(extractSig.InputFields, trajectoryField)

	// Include original outputs
	copy(extractSig.OutputFields, baseSig.OutputFields)

	return extractSig
}

// buildReActInstructions builds the instruction string for the ReAct agent.
func (r *ReAct) buildReActInstructions(baseSig *signatures.Signature, tools map[string]*Tool) string {
	var instructions strings.Builder

	if baseSig.Instructions != "" {
		instructions.WriteString(baseSig.Instructions)
		instructions.WriteString("\n\n")
	}

	instructions.WriteString("You are an Agent. In each episode, you will use tools to collect information.\n")
	instructions.WriteString("Available tools:\n")

	i := 1
	for _, tool := range tools {
		instructions.WriteString(fmt.Sprintf("%d. %s: %s\n", i, tool.Name, tool.Description))
		i++
	}

	instructions.WriteString("\nFor each turn, provide:\n")
	instructions.WriteString("- next_thought: Your reasoning about what to do next\n")
	instructions.WriteString("- next_tool_name: The tool to use\n")
	instructions.WriteString("- next_tool_args: Arguments in JSON format\n")

	return instructions.String()
}

// Forward executes the ReAct reasoning and acting loop.
func (r *ReAct) Forward(ctx context.Context, inputs map[string]interface{}) (*primitives.Prediction, error) {
	trajectory := make(map[string]interface{})
	maxIters := r.MaxIters

	// Allow overriding max iterations
	if maxItersInterface, ok := inputs["max_iters"]; ok {
		if maxItersInt, ok := maxItersInterface.(int); ok {
			maxIters = maxItersInt
			delete(inputs, "max_iters")
		}
	}

	for idx := 0; idx < maxIters; idx++ {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Add trajectory to inputs
		reactInputs := r.copyInputs(inputs)
		reactInputs["trajectory"] = r.formatTrajectory(trajectory)

		// Get next thought, tool, and args
		pred, err := r.ReactModule.Forward(ctx, reactInputs)
		if err != nil {
			return nil, fmt.Errorf("react step %d failed: %w", idx, err)
		}

		outputs := pred.Fields()
		thought := fmt.Sprintf("%v", outputs["next_thought"])
		toolName := fmt.Sprintf("%v", outputs["next_tool_name"])
		toolArgs := outputs["next_tool_args"]

		// Record in trajectory
		trajectory[fmt.Sprintf("thought_%d", idx)] = thought
		trajectory[fmt.Sprintf("tool_name_%d", idx)] = toolName
		trajectory[fmt.Sprintf("tool_args_%d", idx)] = toolArgs

		// Execute the tool
		tool, ok := r.Tools[toolName]
		if !ok {
			observation := fmt.Sprintf("Error: Unknown tool '%s'", toolName)
			trajectory[fmt.Sprintf("observation_%d", idx)] = observation
			continue
		}

		// Convert tool args to map
		argsMap, ok := toolArgs.(map[string]interface{})
		if !ok {
			// Try to convert if it's not already a map
			argsMap = make(map[string]interface{})
		}

		observation, err := tool.Func(argsMap)
		if err != nil {
			observation = fmt.Sprintf("Execution error: %v", err)
		}

		trajectory[fmt.Sprintf("observation_%d", idx)] = observation

		// Check if we're done
		if toolName == "finish" {
			break
		}
	}

	// Extract final answer using the complete trajectory
	extractInputs := r.copyInputs(inputs)
	extractInputs["trajectory"] = r.formatTrajectory(trajectory)

	extract, err := r.ExtractModule.Forward(ctx, extractInputs)
	if err != nil {
		return nil, fmt.Errorf("extraction failed: %w", err)
	}

	// Add trajectory to final prediction
	finalOutputs := extract.Fields()
	finalOutputs["trajectory"] = trajectory

	result := primitives.NewPrediction(finalOutputs)
	result.SetMetadata("react_iterations", len(trajectory)/4) // Each iteration has 4 entries

	return result, nil
}

// formatTrajectory converts the trajectory map to a human-readable string.
func (r *ReAct) formatTrajectory(trajectory map[string]interface{}) string {
	if len(trajectory) == 0 {
		return "No previous actions."
	}

	var formatted strings.Builder
	i := 0
	for {
		thoughtKey := fmt.Sprintf("thought_%d", i)
		if _, ok := trajectory[thoughtKey]; !ok {
			break
		}

		formatted.WriteString(fmt.Sprintf("\nIteration %d:\n", i+1))
		formatted.WriteString(fmt.Sprintf("Thought: %v\n", trajectory[thoughtKey]))
		formatted.WriteString(fmt.Sprintf("Tool: %v\n", trajectory[fmt.Sprintf("tool_name_%d", i)]))
		formatted.WriteString(fmt.Sprintf("Args: %v\n", trajectory[fmt.Sprintf("tool_args_%d", i)]))
		formatted.WriteString(fmt.Sprintf("Observation: %v\n", trajectory[fmt.Sprintf("observation_%d", i)]))

		i++
	}

	return formatted.String()
}

// copyInputs creates a copy of the inputs map.
func (r *ReAct) copyInputs(inputs map[string]interface{}) map[string]interface{} {
	copied := make(map[string]interface{})
	for k, v := range inputs {
		copied[k] = v
	}
	return copied
}

// Copy creates a deep copy of the ReAct module.
func (r *ReAct) Copy() primitives.Module {
	newReAct := &ReAct{
		BaseModule:    primitives.NewBaseModule(),
		Signature:     r.Signature,
		Tools:         r.Tools, // Tools are functions, safe to share
		MaxIters:      r.MaxIters,
		ReactModule:   r.ReactModule.Copy().(*Predict),
		ExtractModule: r.ExtractModule.Copy().(*ChainOfThought),
		Config:        make(map[string]interface{}),
	}

	for k, v := range r.Config {
		newReAct.Config[k] = v
	}

	return newReAct
}

// NamedParameters returns all parameters in this module.
func (r *ReAct) NamedParameters() []primitives.NamedParameter {
	params := r.ReactModule.NamedParameters()
	params = append(params, r.ExtractModule.NamedParameters()...)
	return params
}
