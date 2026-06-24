import sys
from unittest.mock import MagicMock, patch

# Mock confluent_kafka and redis modules before importing worker
class MockKafkaException(Exception):
    pass

class MockKafkaError:
    _PARTITION_EOF = -176
    def __init__(self, code=-176):
        self._code = code
    def code(self):
        return self._code

mock_kafka = MagicMock()
mock_kafka.Consumer = MagicMock()
mock_kafka.KafkaError = MockKafkaError
mock_kafka.KafkaException = MockKafkaException

sys.modules['redis'] = MagicMock()
sys.modules['confluent_kafka'] = mock_kafka

import os
import json
import time
import pytest
import numpy as np
import torch

# Ensure deepvol package is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from deepvol.calibration.worker import CalibrationWorker
from deepvol.calibration.interface import CalibrationResult

class TestCalibrationWorkerEmpirical:
    @pytest.fixture
    def mock_redis_class(self):
        with patch("redis.Redis") as mock:
            redis_instance = MagicMock()
            mock.return_value = redis_instance
            yield redis_instance

    @pytest.fixture
    def mock_kafka_consumer_class(self):
        with patch("deepvol.calibration.worker.Consumer") as mock:
            consumer_instance = MagicMock()
            mock.return_value = consumer_instance
            yield consumer_instance

    def test_worker_initialization(self, mock_redis_class, mock_kafka_consumer_class):
        """Test that CalibrationWorker initializes Kafka and Redis connections with config."""
        worker = CalibrationWorker()
        assert worker.kafka_servers == "localhost:9092"
        assert worker.kafka_topic == "option-quotes"
        assert worker.redis_host == "localhost"
        assert worker.redis_port == 6379

    def test_process_payload_valid_market_iv(self, mock_redis_class, mock_kafka_consumer_class):
        """Test process_payload with a valid 8x11 market_iv grid."""
        worker = CalibrationWorker()
        
        # Mock redis pipeline
        pipe = MagicMock()
        mock_redis_class.pipeline.return_value = pipe
        
        # Valid 8x11 market IV surface
        market_iv = np.full((8, 11), 0.25, dtype=np.float32).tolist()
        payload = {
            "currency": "BTC",
            "model_name": "heston",
            "timestamp": 123456789.0,
            "market_iv": market_iv
        }
        
        # Mock FNO model and calibrate function to prevent real heavy training/optimization
        mock_calib_res = CalibrationResult(
            parameters={"kappa": 1.5, "theta": 0.04, "sigma": 0.3, "rho": -0.5, "v0": 0.04},
            rmse=0.01,
            elapsed_time=0.05,
            status="converged",
            info={"iterations": 10}
        )
        
        with patch("deepvol.calibration.worker.calibrate", return_value=mock_calib_res) as mock_calibrate, \
             patch.object(worker, "get_model", return_value=MagicMock()):
            
            worker.process_payload(payload)
            
            # Assert calibrate was called with the correct args
            mock_calibrate.assert_called_once()
            args, kwargs = mock_calibrate.call_args
            assert kwargs["model_name"] == "heston"
            assert np.allclose(kwargs["market_iv_surface"], np.array(market_iv))
            
            # Assert Redis calls
            mock_redis_class.pipeline.assert_called_once()
            pipe.set.assert_called_once()
            pipe.lpush.assert_called_once()
            pipe.ltrim.assert_called_once()
            pipe.execute.assert_called_once()
            
            # Verify serialization
            latest_key = "deepvol:calibration:latest:btc:heston"
            
            set_call_args = pipe.set.call_args[0]
            assert set_call_args[0] == latest_key
            saved_payload = json.loads(set_call_args[1])
            assert saved_payload["currency"] == "BTC"
            assert saved_payload["model_name"] == "heston"
            assert saved_payload["rmse"] == 0.01
            assert saved_payload["parameters"]["kappa"] == pytest.approx(1.5)

    def test_process_payload_valid_quotes(self, mock_redis_class, mock_kafka_consumer_class):
        """Test process_payload when payload contains 'quotes' that need interpolation."""
        worker = CalibrationWorker()
        
        pipe = MagicMock()
        mock_redis_class.pipeline.return_value = pipe
        
        # Provide some dummy quotes
        payload = {
            "currency": "ETH",
            "model_name": "sabr",
            "timestamp": 123456789.0,
            "quotes": [
                {"strike": 1000.0, "maturity": 0.1, "ask": 0.25, "bid": 0.23, "underlying_price": 1000.0},
                {"strike": 1100.0, "maturity": 0.1, "ask": 0.20, "bid": 0.18, "underlying_price": 1000.0}
            ]
        }
        
        # Mock build_iv_surface to return a valid 8x11 grid
        mock_grid = np.full((8, 11), 0.30, dtype=np.float32)
        mock_calib_res = CalibrationResult(
            parameters=np.array([0.25, -0.4, 0.45], dtype=np.float32),
            rmse=0.02,
            elapsed_time=0.04,
            status="converged",
            info={}
        )
        
        with patch("deepvol.calibration.worker.build_iv_surface", return_value=mock_grid) as mock_build, \
             patch("deepvol.calibration.worker.calibrate", return_value=mock_calib_res) as mock_calibrate, \
             patch.object(worker, "get_model", return_value=MagicMock()):
            
            worker.process_payload(payload)
            mock_build.assert_called_once()
            mock_calibrate.assert_called_once()
            
            # Assert Redis calls
            pipe.set.assert_called_once()
            set_call_args = pipe.set.call_args[0]
            saved_payload = json.loads(set_call_args[1])
            assert saved_payload["currency"] == "ETH"
            assert saved_payload["model_name"] == "sabr"
            # Verify parameters were correctly mapped for SABR
            assert saved_payload["parameters"]["alpha"] == pytest.approx(0.25)
            assert saved_payload["parameters"]["rho"] == pytest.approx(-0.4)
            assert saved_payload["parameters"]["nu"] == pytest.approx(0.45)

    def test_process_payload_missing_data(self, mock_redis_class, mock_kafka_consumer_class):
        """Test process_payload when payload has neither 'market_iv' nor 'quotes'."""
        worker = CalibrationWorker()
        
        payload = {
            "currency": "BTC",
            "model_name": "heston",
            "timestamp": 123456789.0
        }
        
        with patch("deepvol.calibration.worker.calibrate") as mock_calibrate:
            worker.process_payload(payload)
            mock_calibrate.assert_not_called()
            mock_redis_class.pipeline.assert_not_called()

    def test_process_payload_invalid_shape(self, mock_redis_class, mock_kafka_consumer_class):
        """Test process_payload when 'market_iv' has an invalid shape."""
        worker = CalibrationWorker()
        
        payload = {
            "currency": "BTC",
            "model_name": "heston",
            "timestamp": 123456789.0,
            "market_iv": [[0.2, 0.2], [0.3, 0.3]]  # 2x2 instead of 8x11
        }
        
        with patch("deepvol.calibration.worker.calibrate") as mock_calibrate:
            worker.process_payload(payload)
            mock_calibrate.assert_not_called()
            mock_redis_class.pipeline.assert_not_called()

    def test_process_payload_unknown_model(self, mock_redis_class, mock_kafka_consumer_class):
        """Test process_payload with an unknown model name to verify fallback serialization."""
        worker = CalibrationWorker()
        
        pipe = MagicMock()
        mock_redis_class.pipeline.return_value = pipe
        
        market_iv = np.full((8, 11), 0.25, dtype=np.float32).tolist()
        payload = {
            "currency": "BTC",
            "model_name": "unknown_model",
            "timestamp": 123456789.0,
            "market_iv": market_iv
        }
        
        mock_calib_res = CalibrationResult(
            parameters=np.array([1.2, 3.4], dtype=np.float32),
            rmse=0.01,
            elapsed_time=0.05,
            status="converged",
            info={}
        )
        
        with patch("deepvol.calibration.worker.calibrate", return_value=mock_calib_res), \
             patch.object(worker, "get_model", return_value=MagicMock()):
            
            worker.process_payload(payload)
            
            # Assert Redis calls
            pipe.set.assert_called_once()
            set_call_args = pipe.set.call_args[0]
            saved_payload = json.loads(set_call_args[1])
            assert saved_payload["parameters"]["param_0"] == pytest.approx(1.2)
            assert saved_payload["parameters"]["param_1"] == pytest.approx(3.4)
