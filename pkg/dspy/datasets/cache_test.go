package datasets

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGetCacheDir_DefaultTempDir(t *testing.T) {
	// Save original env var
	origDir := os.Getenv("DSPY_CACHE_DIR")
	defer os.Setenv("DSPY_CACHE_DIR", origDir)

	// Test with no env var set
	os.Unsetenv("DSPY_CACHE_DIR")

	cacheDir := getCacheDir("testds")
	expected := filepath.Join(os.TempDir(), "dspy", "testds")

	if cacheDir != expected {
		t.Errorf("expected cache dir %s, got %s", expected, cacheDir)
	}
}

func TestGetCacheDir_WithEnvVar(t *testing.T) {
	// Save original env var
	origDir := os.Getenv("DSPY_CACHE_DIR")
	defer os.Setenv("DSPY_CACHE_DIR", origDir)

	// Test with custom env var
	os.Setenv("DSPY_CACHE_DIR", "/custom/cache")

	cacheDir := getCacheDir("testds")
	expected := filepath.Join("/custom/cache", "dspy", "testds")

	if cacheDir != expected {
		t.Errorf("expected cache dir %s, got %s", expected, cacheDir)
	}
}

func TestNewSQuAD_UsesEnvVar(t *testing.T) {
	// Save original env var
	origDir := os.Getenv("DSPY_CACHE_DIR")
	defer os.Setenv("DSPY_CACHE_DIR", origDir)

	// Set custom cache dir
	os.Setenv("DSPY_CACHE_DIR", "/k8s/cache")

	squad := NewSQuAD(SQuADConfig{})
	expected := filepath.Join("/k8s/cache", "dspy", "squad")

	if squad.cacheDir != expected {
		t.Errorf("expected cache dir %s, got %s", expected, squad.cacheDir)
	}
}

func TestNewGSM8K_UsesEnvVar(t *testing.T) {
	// Save original env var
	origDir := os.Getenv("DSPY_CACHE_DIR")
	defer os.Setenv("DSPY_CACHE_DIR", origDir)

	// Set custom cache dir
	os.Setenv("DSPY_CACHE_DIR", "/docker/cache")

	gsm8k := NewGSM8K(GSM8KConfig{})
	expected := filepath.Join("/docker/cache", "dspy", "gsm8k")

	if gsm8k.cacheDir != expected {
		t.Errorf("expected cache dir %s, got %s", expected, gsm8k.cacheDir)
	}
}

func TestNewHotPotQA_UsesEnvVar(t *testing.T) {
	// Save original env var
	origDir := os.Getenv("DSPY_CACHE_DIR")
	defer os.Setenv("DSPY_CACHE_DIR", origDir)

	// Set custom cache dir
	os.Setenv("DSPY_CACHE_DIR", "/persistent/cache")

	hotpot := NewHotPotQA(HotPotQAConfig{})
	expected := filepath.Join("/persistent/cache", "dspy", "hotpotqa")

	if hotpot.cacheDir != expected {
		t.Errorf("expected cache dir %s, got %s", expected, hotpot.cacheDir)
	}
}

func TestNewMMLU_UsesEnvVar(t *testing.T) {
	// Save original env var
	origDir := os.Getenv("DSPY_CACHE_DIR")
	defer os.Setenv("DSPY_CACHE_DIR", origDir)

	// Set custom cache dir
	os.Setenv("DSPY_CACHE_DIR", "/cache")

	mmlu := NewMMLU(MMLUConfig{})
	expected := filepath.Join("/cache", "dspy", "mmlu")

	if mmlu.cacheDir != expected {
		t.Errorf("expected cache dir %s, got %s", expected, mmlu.cacheDir)
	}
}

func TestNewHellaSwag_UsesEnvVar(t *testing.T) {
	// Save original env var
	origDir := os.Getenv("DSPY_CACHE_DIR")
	defer os.Setenv("DSPY_CACHE_DIR", origDir)

	// Set custom cache dir
	os.Setenv("DSPY_CACHE_DIR", "/data/cache")

	hellaswag := NewHellaSwag(HellaSwagConfig{})
	expected := filepath.Join("/data/cache", "dspy", "hellaswag")

	if hellaswag.cacheDir != expected {
		t.Errorf("expected cache dir %s, got %s", expected, hellaswag.cacheDir)
	}
}

func TestDatasetLoaders_ExplicitCacheDirOverridesEnvVar(t *testing.T) {
	// Save original env var
	origDir := os.Getenv("DSPY_CACHE_DIR")
	defer os.Setenv("DSPY_CACHE_DIR", origDir)

	// Set custom env var
	os.Setenv("DSPY_CACHE_DIR", "/env/cache")

	// Explicit cache dir should override env var
	squad := NewSQuAD(SQuADConfig{CacheDir: "/explicit/cache"})
	if squad.cacheDir != "/explicit/cache" {
		t.Errorf("expected explicit cache dir to override env var, got %s", squad.cacheDir)
	}
}
