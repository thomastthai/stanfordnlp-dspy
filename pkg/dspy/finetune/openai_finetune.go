package finetune

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"time"
)

// OpenAIFineTuner implements fine-tuning for OpenAI models.
type OpenAIFineTuner struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

// NewOpenAIFineTuner creates a new OpenAI fine-tuner.
func NewOpenAIFineTuner(apiKey string) *OpenAIFineTuner {
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	
	return &OpenAIFineTuner{
		apiKey:  apiKey,
		baseURL: "https://api.openai.com/v1",
		client:  &http.Client{},
	}
}

// PrepareData implements FineTuner.PrepareData.
func (f *OpenAIFineTuner) PrepareData(examples []Example) (string, error) {
	file, err := os.CreateTemp("", "finetune-*.jsonl")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer file.Close()
	
	for _, ex := range examples {
		data, err := json.Marshal(ex)
		if err != nil {
			return "", fmt.Errorf("failed to marshal example: %w", err)
		}
		if _, err := file.Write(append(data, '\n')); err != nil {
			return "", fmt.Errorf("failed to write example: %w", err)
		}
	}
	
	return file.Name(), nil
}

// StartJob implements FineTuner.StartJob.
func (f *OpenAIFineTuner) StartJob(ctx context.Context, config FineTuneConfig) (string, error) {
	// Upload training file
	fileID, err := f.uploadFile(ctx, config.TrainingFile, "fine-tune")
	if err != nil {
		return "", fmt.Errorf("failed to upload training file: %w", err)
	}
	
	// Create fine-tuning job
	reqBody := map[string]interface{}{
		"training_file": fileID,
		"model":         config.BaseModel,
	}
	
	if config.ValidationFile != "" {
		valFileID, err := f.uploadFile(ctx, config.ValidationFile, "fine-tune")
		if err != nil {
			return "", fmt.Errorf("failed to upload validation file: %w", err)
		}
		reqBody["validation_file"] = valFileID
	}
	
	if config.Epochs > 0 {
		reqBody["hyperparameters"] = map[string]interface{}{
			"n_epochs": config.Epochs,
		}
	}
	
	if config.Suffix != "" {
		reqBody["suffix"] = config.Suffix
	}
	
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}
	
	url := fmt.Sprintf("%s/fine_tuning/jobs", f.baseURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", f.apiKey))
	
	resp, err := f.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()
	
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API request failed: %s", string(respBody))
	}
	
	var result struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}
	
	return result.ID, nil
}

// GetStatus implements FineTuner.GetStatus.
func (f *OpenAIFineTuner) GetStatus(ctx context.Context, jobID string) (JobStatus, error) {
	url := fmt.Sprintf("%s/fine_tuning/jobs/%s", f.baseURL, jobID)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return JobStatus{}, fmt.Errorf("failed to create request: %w", err)
	}
	
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", f.apiKey))
	
	resp, err := f.client.Do(req)
	if err != nil {
		return JobStatus{}, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()
	
	var result struct {
		ID           string `json:"id"`
		Status       string `json:"status"`
		CreatedAt    int64  `json:"created_at"`
		FinishedAt   int64  `json:"finished_at"`
		FineTunedModel string `json:"fine_tuned_model"`
		TrainedTokens int   `json:"trained_tokens"`
		Error        struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return JobStatus{}, fmt.Errorf("failed to parse response: %w", err)
	}
	
	status := JobStatus{
		ID:           result.ID,
		Status:       result.Status,
		Model:        result.FineTunedModel,
		TrainedTokens: result.TrainedTokens,
		Error:        result.Error.Message,
	}
	
	if result.CreatedAt > 0 {
		status.CreatedAt = time.Unix(result.CreatedAt, 0)
	}
	if result.FinishedAt > 0 {
		status.FinishedAt = time.Unix(result.FinishedAt, 0)
	}
	
	return status, nil
}

// Cancel implements FineTuner.Cancel.
func (f *OpenAIFineTuner) Cancel(ctx context.Context, jobID string) error {
	url := fmt.Sprintf("%s/fine_tuning/jobs/%s/cancel", f.baseURL, jobID)
	req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", f.apiKey))
	
	resp, err := f.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to cancel job: status %d", resp.StatusCode)
	}
	
	return nil
}

// GetModel implements FineTuner.GetModel.
func (f *OpenAIFineTuner) GetModel(ctx context.Context, jobID string) (string, error) {
	status, err := f.GetStatus(ctx, jobID)
	if err != nil {
		return "", err
	}
	
	if status.Model == "" {
		return "", fmt.Errorf("model not yet available")
	}
	
	return status.Model, nil
}

// uploadFile uploads a file to OpenAI.
func (f *OpenAIFineTuner) uploadFile(ctx context.Context, path string, purpose string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()
	
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	
	part, err := writer.CreateFormFile("file", path)
	if err != nil {
		return "", fmt.Errorf("failed to create form file: %w", err)
	}
	
	if _, err := io.Copy(part, file); err != nil {
		return "", fmt.Errorf("failed to copy file: %w", err)
	}
	
	if err := writer.WriteField("purpose", purpose); err != nil {
		return "", fmt.Errorf("failed to write purpose: %w", err)
	}
	
	if err := writer.Close(); err != nil {
		return "", fmt.Errorf("failed to close writer: %w", err)
	}
	
	url := fmt.Sprintf("%s/files", f.baseURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, body)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", f.apiKey))
	
	resp, err := f.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()
	
	var result struct {
		ID string `json:"id"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}
	
	return result.ID, nil
}
