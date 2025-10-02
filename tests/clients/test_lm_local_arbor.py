import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from datetime import datetime
import requests

from dspy.clients.lm_local_arbor import (
    ArborProvider, 
    ArborTrainingJob, 
    ArborReinforceJob
)
from dspy.clients.utils_finetune import (
    TrainingStatus, 
    TrainDataFormat, 
    GRPOGroup,
    GRPOChatData,
    MultiGPUConfig
)
from dspy.teleprompt.grpo.grpo_config import GRPOConfig
from dspy.clients.lm import LM


# Fixtures
@pytest.fixture
def mock_lm():
    lm = Mock(spec=LM)
    lm.model = "openai/arbor:test-model"
    lm.kwargs = {
        "api_base": "http://localhost:8000/v1/",
        "api_key": "test-key"
    }
    lm.launch_kwargs = {}
    return lm


@pytest.fixture
def grpo_config():
    """Create a basic GRPOConfig for testing."""
    return GRPOConfig(
        num_generations=4,
        temperature=0.9,
        beta=0.04,
        per_device_train_batch_size=2,
        learning_rate=1e-5
    )


@pytest.fixture
def gpu_config():
    return MultiGPUConfig(num_inference_gpus=1, num_training_gpus=1)


class TestArborTrainingJob:
    def test_init(self):
        """Test ArborTrainingJob initialization."""
        job = ArborTrainingJob()
        assert job.provider_file_id is None
        assert job.provider_job_id is None
    
    @patch('dspy.clients.lm_local_arbor.openai.fine_tuning.jobs.cancel')
    @patch('dspy.clients.lm_local_arbor.openai.files.delete')
    @patch.object(ArborProvider, 'does_job_exist', return_value=True)
    @patch.object(ArborProvider, 'does_file_exist', return_value=True)
    @patch.object(ArborProvider, 'is_terminal_training_status', return_value=False)
    def test_cancel_active_job(self, mock_terminal, mock_file_exists, mock_job_exists, 
                                mock_file_delete, mock_job_cancel):
        """Test canceling an active training job."""
        job = ArborTrainingJob()
        job.provider_job_id = "test-job-id" 
        job.provider_file_id = "test-file-id" 
        
        with patch.object(job, 'status', return_value=TrainingStatus.running):
            job.cancel()
        
        mock_job_cancel.assert_called_once_with("test-job-id")
        mock_file_delete.assert_called_once_with("test-file-id")
        assert job.provider_job_id is None
        assert job.provider_file_id is None
    
    @patch.object(ArborProvider, 'get_training_status', return_value=TrainingStatus.running)
    def test_status(self, mock_get_status):
        """Test getting job status."""
        job = ArborTrainingJob()
        job.provider_job_id = "test-job-id"
        
        status = job.status()
        
        assert status == TrainingStatus.running
        mock_get_status.assert_called_once_with("test-job-id")


class TestArborReinforceJob:
    def test_init_with_grpo_config(self, mock_lm, grpo_config, gpu_config):
        """Test ArborReinforceJob initialization with GRPOConfig."""
        job = ArborReinforceJob(mock_lm, grpo_config, gpu_config)
        
        assert job.lm == mock_lm
        assert isinstance(job.config, GRPOConfig)
        
        assert job.provider_job_id is None
        assert job.checkpoints == {}
        assert job.last_checkpoint is None
    
    def test_init_with_invalid_train_kwargs(self, mock_lm, gpu_config):
        """Test that init raises TypeError with non-GRPOConfig"""
        with pytest.raises(TypeError, match="Expected config to be of type GRPOConfig"):
            ArborReinforceJob(mock_lm, {"invalid": "dict"}, gpu_config)
    
    @patch('requests.post')
    def test_initialize(self, mock_post, mock_lm, grpo_config, gpu_config):
        """Test GRPO job initialization."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "current_model": "test-model-v1",
            "job_id": "grpo-job-123"
        }
        mock_post.return_value = mock_response
        
        job = ArborReinforceJob(mock_lm, grpo_config, gpu_config)
        job.initialize()
        
        assert job.provider_job_id == "grpo-job-123"
        assert job.lm.model == "openai/arbor:test-model-v1"
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_run_grpo_step_one_group(self, mock_post, mock_lm, grpo_config, gpu_config):
        """Test running a single GRPO step."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"current_model": "test-model-v2"}
        mock_post.return_value = mock_response
        
        job = ArborReinforceJob(mock_lm, grpo_config, gpu_config)
        job.provider_job_id = "grpo-job-123"
        
        train_group: GRPOGroup = [
            GRPOChatData(
                messages=[{"role": "user", "content": "test"}],
                completion={"role": "assistant", "content": "response"},
                reward=1.0
            )
        ]
        
        job._run_grpo_step_one_group(train_group)
        
        assert job.lm.model == "openai/arbor:test-model-v2"
        mock_post.assert_called_once()
    
    def test_step(self, mock_lm, grpo_config, gpu_config):
        """Test step method with GRPO_CHAT format."""
        job = ArborReinforceJob(mock_lm, grpo_config, gpu_config)
        job.provider_job_id = "grpo-job-123"
        
        train_data: list[GRPOGroup] = [
            [
                GRPOChatData(
                    messages=[{"role": "user", "content": "test"}],
                    completion={"role": "assistant", "content": "response"},
                    reward=1.0
                )
            ]
        ]
        
        with patch.object(job, '_run_grpo_step_one_group') as mock_step:
            job.step(train_data, TrainDataFormat.GRPO_CHAT)
            mock_step.assert_called_once()
    
    def test_step_invalid_format(self, mock_lm, grpo_config, gpu_config):
        """Test that step raises error with invalid format."""
        job = ArborReinforceJob(mock_lm, grpo_config, gpu_config)
        
        with pytest.raises(AssertionError):
            job.step([], TrainDataFormat.CHAT)
    
    @patch('requests.post')
    def test_save_checkpoint(self, mock_post, mock_lm, grpo_config, gpu_config):
        """Test saving a checkpoint."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "last_checkpoint": "checkpoint-1",
            "checkpoints": {"checkpoint-1": "/path/to/checkpoint"}
        }
        mock_post.return_value = mock_response
        
        job = ArborReinforceJob(mock_lm, grpo_config, gpu_config)
        job.provider_job_id = "grpo-job-123"
        
        job.save_checkpoint("checkpoint-1", score=0.95)
        
        assert job.last_checkpoint == "checkpoint-1"
        assert "checkpoint-1" in job.checkpoints
        assert job.checkpoints["checkpoint-1"]["score"] == 0.95
    
    @patch('requests.post')
    def test_terminate(self, mock_post, mock_lm, grpo_config, gpu_config):
        """Test terminating GRPO job."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"current_model": "final-model"}
        mock_post.return_value = mock_response
        
        job = ArborReinforceJob(mock_lm, grpo_config, gpu_config)
        job.provider_job_id = "grpo-job-123"
        
        job.terminate()
        
        assert job.lm.model == "openai/arbor:final-model"
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_cancel(self, mock_post, mock_lm, grpo_config, gpu_config):
        """Test canceling GRPO job."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        job = ArborReinforceJob(mock_lm, grpo_config, gpu_config)
        job.provider_job_id = "grpo-job-123"
        
        job.cancel()
        
        assert job.provider_job_id is None
        mock_post.assert_called_once()


# Test ArborProvider
class TestArborProvider:
    def test_init(self):
        """Test ArborProvider initialization."""
        provider = ArborProvider()
        assert provider.finetunable is True
        assert provider.reinforceable is True
        assert provider.TrainingJob == ArborTrainingJob
        assert provider.ReinforceJob == ArborReinforceJob
    
    @patch('requests.post')
    def test_launch(self, mock_post, mock_lm):
        """Test launching inference server."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        ArborProvider.launch(mock_lm)
        
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "chat/launch" in call_args.args[0]
    
    @patch('requests.post')
    def test_kill(self, mock_post, mock_lm):
        """Test killing inference server."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        ArborProvider.kill(mock_lm)
        
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "chat/kill" in call_args.args[0]
    
    def test_remove_provider_prefix(self):
        """Test removing provider prefix from model name."""
        assert ArborProvider._remove_provider_prefix("openai/arbor:model") == "model"
        assert ArborProvider._remove_provider_prefix("arbor:model") == "model"
        assert ArborProvider._remove_provider_prefix("model") == "model"
    
    def test_add_provider_prefix(self):
        """Test adding provider prefix to model name."""
        assert ArborProvider._add_provider_prefix("model") == "openai/arbor:model"
        assert ArborProvider._add_provider_prefix("openai/arbor:model") == "openai/arbor:model"
    
    def test_validate_data_format_valid(self):
        """Test data format validation with valid formats."""
        ArborProvider.validate_data_format(TrainDataFormat.CHAT)
        ArborProvider.validate_data_format(TrainDataFormat.COMPLETION)
        ArborProvider.validate_data_format(TrainDataFormat.GRPO_CHAT)
    
    def test_validate_data_format_invalid(self):
        """Test data format validation with invalid format."""
        with pytest.raises(ValueError, match="does not support"):
            ArborProvider.validate_data_format("invalid_format")  # type: ignore
    
    @patch('dspy.clients.lm_local_arbor.openai.files.create')
    @patch('builtins.open', new_callable=mock_open, read_data="data")
    @patch.object(ArborProvider, '_get_arbor_base_api', return_value="http://localhost:8000/v1/")
    def test_upload_data(self, mock_api, mock_file, mock_create):
        """Test uploading data file."""
        mock_file_obj = Mock()
        mock_file_obj.id = "file-123"
        mock_create.return_value = mock_file_obj
        
        file_id = ArborProvider.upload_data("/path/to/data.jsonl", {})
        
        assert file_id == "file-123"
        mock_create.assert_called_once()
    
    def test_is_terminal_training_status(self):
        """Test terminal status detection."""
        assert ArborProvider.is_terminal_training_status(TrainingStatus.succeeded) is True
        assert ArborProvider.is_terminal_training_status(TrainingStatus.failed) is True
        assert ArborProvider.is_terminal_training_status(TrainingStatus.cancelled) is True
        assert ArborProvider.is_terminal_training_status(TrainingStatus.running) is False
        assert ArborProvider.is_terminal_training_status(TrainingStatus.pending) is False
    
    @patch('dspy.clients.lm_local_arbor.openai.fine_tuning.jobs.retrieve')
    @patch.object(ArborProvider, '_get_arbor_base_api', return_value="http://localhost:8000/v1/")
    @patch.object(ArborProvider, 'does_job_exist', return_value=True)
    def test_get_training_status(self, mock_does_job_exist, mock_api, mock_retrieve):
        """Test getting training status."""
        # Reset the mock to ensure clean state
        mock_retrieve.reset_mock()
        
        mock_job = Mock()
        mock_job.status = "running"
        mock_retrieve.return_value = mock_job
        
        status = ArborProvider.get_training_status("job-123", {})
        
        assert status == TrainingStatus.running
        mock_retrieve.assert_called_once_with("job-123")
        mock_does_job_exist.assert_called_once_with("job-123", {})
    
    def test_get_training_status_no_job(self):
        """Test getting status when no job exists."""
        status = ArborProvider.get_training_status(None, {})  # type: ignore
        assert status == TrainingStatus.not_started


# Integration-style tests (still mocked but testing workflows)
class TestArborWorkflows:
    @patch('requests.post')
    @patch('dspy.clients.lm_local_arbor.openai.fine_tuning.jobs.retrieve')
    @patch.object(ArborProvider, '_get_arbor_base_api', return_value="http://localhost:8000/v1/")
    def test_grpo_workflow(self, mock_api, mock_retrieve, mock_post, mock_lm, grpo_config, gpu_config):
        """Test a complete GRPO workflow."""
        # Reset mocks to ensure clean state
        mock_retrieve.reset_mock()
        mock_post.reset_mock()
        
        # Mock responses
        init_response = Mock()
        init_response.status_code = 200
        init_response.json.return_value = {
            "current_model": "model-v1",
            "job_id": "grpo-123"
        }
        
        step_response = Mock()
        step_response.status_code = 200
        step_response.json.return_value = {"current_model": "model-v2"}
        
        checkpoint_response = Mock()
        checkpoint_response.status_code = 200
        checkpoint_response.json.return_value = {
            "last_checkpoint": "ckpt-1",
            "checkpoints": {"ckpt-1": "/path"}
        }
        
        terminate_response = Mock()
        terminate_response.status_code = 200
        terminate_response.json.return_value = {"current_model": "final-model"}
        
        mock_post.side_effect = [init_response, step_response, checkpoint_response, terminate_response]
        
        # Create and initialize job
        job = ArborReinforceJob(mock_lm, grpo_config, gpu_config)
        job.initialize()
        
        assert job.provider_job_id == "grpo-123"
        
        # Run a step
        train_group: GRPOGroup = [
            GRPOChatData(
                messages=[{"role": "user", "content": "test"}],
                completion={"role": "assistant", "content": "response"},
                reward=1.0
            )
        ]
        job._run_grpo_step_one_group(train_group)
        
        job.save_checkpoint("ckpt-1", score=0.9)
        
        assert job.last_checkpoint == "ckpt-1"
        
        # Terminate
        job.terminate()
        
        assert job.lm.model == "openai/arbor:final-model"
        assert mock_post.call_count == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])