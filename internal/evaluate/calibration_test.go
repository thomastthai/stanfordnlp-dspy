package evaluate

import (
	"math"
	"testing"
)

func TestMetricCalibrator_ComputeCorrelation(t *testing.T) {
	calibrator := NewMetricCalibrator()

	// Add perfectly correlated data
	calibrator.AddPair("ex1", 0.8, 0.8)
	calibrator.AddPair("ex2", 0.6, 0.6)
	calibrator.AddPair("ex3", 0.9, 0.9)

	correlation, err := calibrator.ComputeCorrelation()
	if err != nil {
		t.Fatalf("ComputeCorrelation failed: %v", err)
	}

	// Perfect correlation should be 1.0
	if math.Abs(correlation-1.0) > 0.001 {
		t.Errorf("Expected correlation ~1.0, got %f", correlation)
	}
}

func TestMetricCalibrator_ComputeAgreement(t *testing.T) {
	calibrator := NewMetricCalibrator()

	// Add data with 75% agreement
	calibrator.AddPair("ex1", 0.8, 0.7) // both >= 0.5, agree
	calibrator.AddPair("ex2", 0.3, 0.2) // both < 0.5, agree
	calibrator.AddPair("ex3", 0.6, 0.4) // disagree
	calibrator.AddPair("ex4", 0.9, 0.8) // both >= 0.5, agree

	agreement, err := calibrator.ComputeAgreement()
	if err != nil {
		t.Fatalf("ComputeAgreement failed: %v", err)
	}

	expected := 0.75
	if math.Abs(agreement-expected) > 0.001 {
		t.Errorf("Expected agreement %f, got %f", expected, agreement)
	}
}

func TestMetricCalibrator_ComputeMAE(t *testing.T) {
	calibrator := NewMetricCalibrator()

	calibrator.AddPair("ex1", 0.8, 0.7) // error: 0.1
	calibrator.AddPair("ex2", 0.6, 0.8) // error: 0.2
	calibrator.AddPair("ex3", 0.9, 0.9) // error: 0.0

	mae, err := calibrator.ComputeMAE()
	if err != nil {
		t.Fatalf("ComputeMAE failed: %v", err)
	}

	expected := 0.1 // (0.1 + 0.2 + 0.0) / 3
	if math.Abs(mae-expected) > 0.001 {
		t.Errorf("Expected MAE %f, got %f", expected, mae)
	}
}

func TestMetricCalibrator_ComputeRMSE(t *testing.T) {
	calibrator := NewMetricCalibrator()

	calibrator.AddPair("ex1", 0.8, 0.7) // squared error: 0.01
	calibrator.AddPair("ex2", 0.6, 0.8) // squared error: 0.04
	calibrator.AddPair("ex3", 0.9, 0.9) // squared error: 0.00

	rmse, err := calibrator.ComputeRMSE()
	if err != nil {
		t.Fatalf("ComputeRMSE failed: %v", err)
	}

	expected := math.Sqrt(0.05 / 3) // sqrt((0.01 + 0.04 + 0.00) / 3)
	if math.Abs(rmse-expected) > 0.001 {
		t.Errorf("Expected RMSE %f, got %f", expected, rmse)
	}
}

func TestMetricCalibrator_DetectBias(t *testing.T) {
	calibrator := NewMetricCalibrator()

	// Add data with systematic over-estimation
	calibrator.AddPair("ex1", 0.5, 0.7) // +0.2
	calibrator.AddPair("ex2", 0.6, 0.8) // +0.2
	calibrator.AddPair("ex3", 0.7, 0.9) // +0.2

	biases, err := calibrator.DetectBias()
	if err != nil {
		t.Fatalf("DetectBias failed: %v", err)
	}

	if _, ok := biases["mean_bias"]; !ok {
		t.Error("Expected mean_bias in results")
	}

	if _, ok := biases["over_estimation"]; !ok {
		t.Error("Expected over_estimation in results")
	}

	meanBias := biases["mean_bias"]
	expected := 0.2
	if math.Abs(meanBias-expected) > 0.001 {
		t.Errorf("Expected mean bias %f, got %f", expected, meanBias)
	}
}

func TestMetricCalibrator_GenerateReport(t *testing.T) {
	calibrator := NewMetricCalibrator()

	// Add sample data
	calibrator.AddPair("ex1", 0.8, 0.75)
	calibrator.AddPair("ex2", 0.6, 0.65)
	calibrator.AddPair("ex3", 0.9, 0.85)
	calibrator.AddPair("ex4", 0.7, 0.7)

	report, err := calibrator.GenerateReport()
	if err != nil {
		t.Fatalf("GenerateReport failed: %v", err)
	}

	if report.NumSamples != 4 {
		t.Errorf("Expected 4 samples, got %d", report.NumSamples)
	}

	if report.Correlation < 0 || report.Correlation > 1 {
		t.Errorf("Correlation should be between 0 and 1, got %f", report.Correlation)
	}

	if report.Agreement < 0 || report.Agreement > 1 {
		t.Errorf("Agreement should be between 0 and 1, got %f", report.Agreement)
	}

	if report.ReliabilityNote == "" {
		t.Error("Expected non-empty reliability note")
	}

	// Test String method
	reportStr := report.String()
	if reportStr == "" {
		t.Error("Expected non-empty report string")
	}
}

func TestMetricCalibrator_InsufficientData(t *testing.T) {
	calibrator := NewMetricCalibrator()

	// No data
	_, err := calibrator.ComputeCorrelation()
	if err == nil {
		t.Error("Expected error with no data")
	}

	// Only one pair
	calibrator.AddPair("ex1", 0.8, 0.7)
	_, err = calibrator.ComputeCorrelation()
	if err == nil {
		t.Error("Expected error with only one pair")
	}
}

func TestVariance(t *testing.T) {
	values := []float64{2, 4, 4, 4, 5, 5, 7, 9}
	result := variance(values)

	// Expected variance: 4
	expected := 4.0
	if math.Abs(result-expected) > 0.001 {
		t.Errorf("Expected variance %f, got %f", expected, result)
	}
}

func TestVariance_EmptySlice(t *testing.T) {
	values := []float64{}
	result := variance(values)

	if result != 0.0 {
		t.Errorf("Expected variance 0.0 for empty slice, got %f", result)
	}
}
