package evaluate

import (
	"fmt"
	"math"
)

// MetricCalibrator helps assess the reliability of evaluation metrics.
type MetricCalibrator struct {
	HumanLabels     map[string]float64
	PredictedScores map[string]float64
}

// NewMetricCalibrator creates a new calibrator for metric assessment.
func NewMetricCalibrator() *MetricCalibrator {
	return &MetricCalibrator{
		HumanLabels:     make(map[string]float64),
		PredictedScores: make(map[string]float64),
	}
}

// AddPair adds a human label and predicted score pair for an example.
func (m *MetricCalibrator) AddPair(exampleID string, humanLabel, predictedScore float64) {
	m.HumanLabels[exampleID] = humanLabel
	m.PredictedScores[exampleID] = predictedScore
}

// ComputeCorrelation calculates the Pearson correlation coefficient between human labels and predicted scores.
func (m *MetricCalibrator) ComputeCorrelation() (float64, error) {
	if len(m.HumanLabels) == 0 || len(m.PredictedScores) == 0 {
		return 0.0, fmt.Errorf("insufficient data for correlation computation")
	}

	// Ensure we have matching pairs
	var humanVals, predictedVals []float64
	for id, humanLabel := range m.HumanLabels {
		if predictedScore, ok := m.PredictedScores[id]; ok {
			humanVals = append(humanVals, humanLabel)
			predictedVals = append(predictedVals, predictedScore)
		}
	}

	if len(humanVals) < 2 {
		return 0.0, fmt.Errorf("need at least 2 matching pairs for correlation")
	}

	// Calculate means
	humanMean := mean(humanVals)
	predictedMean := mean(predictedVals)

	// Calculate correlation
	var numerator, humanSumSq, predictedSumSq float64
	for i := 0; i < len(humanVals); i++ {
		humanDiff := humanVals[i] - humanMean
		predictedDiff := predictedVals[i] - predictedMean

		numerator += humanDiff * predictedDiff
		humanSumSq += humanDiff * humanDiff
		predictedSumSq += predictedDiff * predictedDiff
	}

	denominator := math.Sqrt(humanSumSq * predictedSumSq)
	if denominator == 0 {
		return 0.0, nil
	}

	return numerator / denominator, nil
}

// ComputeAgreement calculates the agreement rate between human and predicted binary judgments.
// Assumes scores >= 0.5 are positive, < 0.5 are negative.
func (m *MetricCalibrator) ComputeAgreement() (float64, error) {
	if len(m.HumanLabels) == 0 || len(m.PredictedScores) == 0 {
		return 0.0, fmt.Errorf("insufficient data for agreement computation")
	}

	agreements := 0
	total := 0

	for id, humanLabel := range m.HumanLabels {
		if predictedScore, ok := m.PredictedScores[id]; ok {
			humanBinary := humanLabel >= 0.5
			predictedBinary := predictedScore >= 0.5

			if humanBinary == predictedBinary {
				agreements++
			}
			total++
		}
	}

	if total == 0 {
		return 0.0, fmt.Errorf("no matching pairs found")
	}

	return float64(agreements) / float64(total), nil
}

// ComputeMAE calculates the Mean Absolute Error between human labels and predicted scores.
func (m *MetricCalibrator) ComputeMAE() (float64, error) {
	if len(m.HumanLabels) == 0 || len(m.PredictedScores) == 0 {
		return 0.0, fmt.Errorf("insufficient data for MAE computation")
	}

	var sumAbsError float64
	count := 0

	for id, humanLabel := range m.HumanLabels {
		if predictedScore, ok := m.PredictedScores[id]; ok {
			absError := math.Abs(humanLabel - predictedScore)
			sumAbsError += absError
			count++
		}
	}

	if count == 0 {
		return 0.0, fmt.Errorf("no matching pairs found")
	}

	return sumAbsError / float64(count), nil
}

// ComputeRMSE calculates the Root Mean Square Error between human labels and predicted scores.
func (m *MetricCalibrator) ComputeRMSE() (float64, error) {
	if len(m.HumanLabels) == 0 || len(m.PredictedScores) == 0 {
		return 0.0, fmt.Errorf("insufficient data for RMSE computation")
	}

	var sumSquaredError float64
	count := 0

	for id, humanLabel := range m.HumanLabels {
		if predictedScore, ok := m.PredictedScores[id]; ok {
			error := humanLabel - predictedScore
			sumSquaredError += error * error
			count++
		}
	}

	if count == 0 {
		return 0.0, fmt.Errorf("no matching pairs found")
	}

	return math.Sqrt(sumSquaredError / float64(count)), nil
}

// DetectBias analyzes potential biases in the metric.
// Returns a map of bias types to their severity scores.
func (m *MetricCalibrator) DetectBias() (map[string]float64, error) {
	if len(m.HumanLabels) == 0 || len(m.PredictedScores) == 0 {
		return nil, fmt.Errorf("insufficient data for bias detection")
	}

	biases := make(map[string]float64)

	// Compute mean bias (predicted - human)
	var sumBias float64
	count := 0

	for id, humanLabel := range m.HumanLabels {
		if predictedScore, ok := m.PredictedScores[id]; ok {
			bias := predictedScore - humanLabel
			sumBias += bias
			count++
		}
	}

	if count > 0 {
		meanBias := sumBias / float64(count)
		biases["mean_bias"] = meanBias

		// Positive bias indicates over-estimation
		if meanBias > 0.1 {
			biases["over_estimation"] = meanBias
		} else if meanBias < -0.1 {
			biases["under_estimation"] = -meanBias
		}
	}

	// Compute variance bias (differences in spread)
	humanVals := []float64{}
	predictedVals := []float64{}
	for id, humanLabel := range m.HumanLabels {
		if predictedScore, ok := m.PredictedScores[id]; ok {
			humanVals = append(humanVals, humanLabel)
			predictedVals = append(predictedVals, predictedScore)
		}
	}

	if len(humanVals) > 1 {
		humanVar := variance(humanVals)
		predictedVar := variance(predictedVals)

		varianceDiff := math.Abs(humanVar - predictedVar)
		biases["variance_difference"] = varianceDiff

		if predictedVar < humanVar*0.5 {
			biases["under_confidence"] = humanVar - predictedVar
		} else if predictedVar > humanVar*1.5 {
			biases["over_confidence"] = predictedVar - humanVar
		}
	}

	return biases, nil
}

// variance calculates the variance of a slice of values.
func variance(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	m := mean(values)
	var sumSq float64

	for _, v := range values {
		diff := v - m
		sumSq += diff * diff
	}

	return sumSq / float64(len(values))
}

// CalibrationReport generates a comprehensive calibration report.
type CalibrationReport struct {
	Correlation     float64
	Agreement       float64
	MAE             float64
	RMSE            float64
	Biases          map[string]float64
	NumSamples      int
	ReliabilityNote string
}

// GenerateReport creates a comprehensive calibration report.
func (m *MetricCalibrator) GenerateReport() (*CalibrationReport, error) {
	report := &CalibrationReport{
		Biases: make(map[string]float64),
	}

	// Count matching pairs
	for id := range m.HumanLabels {
		if _, ok := m.PredictedScores[id]; ok {
			report.NumSamples++
		}
	}

	if report.NumSamples < 2 {
		return nil, fmt.Errorf("need at least 2 samples for calibration report")
	}

	// Compute metrics
	correlation, err := m.ComputeCorrelation()
	if err != nil {
		return nil, fmt.Errorf("failed to compute correlation: %w", err)
	}
	report.Correlation = correlation

	agreement, err := m.ComputeAgreement()
	if err != nil {
		return nil, fmt.Errorf("failed to compute agreement: %w", err)
	}
	report.Agreement = agreement

	mae, err := m.ComputeMAE()
	if err != nil {
		return nil, fmt.Errorf("failed to compute MAE: %w", err)
	}
	report.MAE = mae

	rmse, err := m.ComputeRMSE()
	if err != nil {
		return nil, fmt.Errorf("failed to compute RMSE: %w", err)
	}
	report.RMSE = rmse

	biases, err := m.DetectBias()
	if err != nil {
		return nil, fmt.Errorf("failed to detect bias: %w", err)
	}
	report.Biases = biases

	// Add reliability note
	if report.Correlation > 0.8 {
		report.ReliabilityNote = "High reliability - strong correlation with human judgments"
	} else if report.Correlation > 0.6 {
		report.ReliabilityNote = "Moderate reliability - reasonable correlation with human judgments"
	} else if report.Correlation > 0.4 {
		report.ReliabilityNote = "Low reliability - weak correlation with human judgments"
	} else {
		report.ReliabilityNote = "Poor reliability - very weak or no correlation with human judgments"
	}

	return report, nil
}

// String returns a formatted string representation of the calibration report.
func (r *CalibrationReport) String() string {
	return fmt.Sprintf(`Calibration Report
==================
Samples: %d
Correlation: %.3f
Agreement: %.3f
MAE: %.3f
RMSE: %.3f
Biases: %v
Reliability: %s`,
		r.NumSamples,
		r.Correlation,
		r.Agreement,
		r.MAE,
		r.RMSE,
		r.Biases,
		r.ReliabilityNote,
	)
}
