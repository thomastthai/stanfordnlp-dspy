package main

import (
	"fmt"
	"os"

	"github.com/stanfordnlp/dspy/pkg/dspy"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "version" {
		fmt.Printf("DSPy-Go v%s (Python compatible: v%s)\n", dspy.Version, dspy.PythonCompatVersion)
		return
	}

	fmt.Println("DSPy-Go CLI")
	fmt.Println("===========")
	fmt.Printf("Version: %s\n", dspy.Version)
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  dspy version    Show version information")
	fmt.Println()
	fmt.Println("For more information, visit: https://dspy.ai")
}
