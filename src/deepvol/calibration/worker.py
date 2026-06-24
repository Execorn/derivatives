"""
deepvol.calibration.worker — GPU-Accelerated Calibration Stream Consumer.

Subscribes to Kafka 'option-quotes', runs calibrations utilizing pre-existing 
surrogate models, and writes the output parameters to Redis.
"""
from __future__ import annotations

import os
import sys
import json
import signal
import logging
import time
from typing import Dict, Any, List, Union

import numpy as np
import torch
import redis
from confluent_kafka import Consumer, KafkaError, KafkaException

# Ensure deepvol package is importable if run directly
_src = str(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
if _src not in sys.path:
    sys.path.insert(0, _src)

from deepvol.calibration.interface import calibrate, CalibrationResult
from deepvol.market.deribit_data import build_iv_surface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("deepvol.worker")


class CalibrationWorker:
    def __init__(self):
        self.running = True
        
        # Load environment configurations
        self.kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.kafka_topic = os.getenv("KAFKA_TOPIC", "option-quotes")
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Redis connection
        logger.info(f"Connecting to Redis Cache at {self.redis_host}:{self.redis_port}")
        self.redis_client = redis.Redis(
            host=self.redis_host, 
            port=self.redis_port, 
            decode_responses=True
        )
        
        # Initialize Kafka Consumer
        # cooperative-sticky is used for minimal rebalance disruption
        self.consumer_conf = {
            "bootstrap.servers": self.kafka_servers,
            "group.id": "deepvol-calibrator-group",
            "auto.offset.reset": os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest"),
            "enable.auto.commit": os.getenv("KAFKA_ENABLE_AUTO_COMMIT", "false").lower() in ("true", "1"),
            "max.poll.interval.ms": int(os.getenv("KAFKA_MAX_POLL_INTERVAL_MS", "300000")),
            "session.timeout.ms": int(os.getenv("KAFKA_SESSION_TIMEOUT_MS", "30000")),
            "partition.assignment.strategy": os.getenv(
                "KAFKA_PARTITION_ASSIGNMENT_STRATEGY", 
                "cooperative-sticky"
            ),
        }
        logger.info(f"Connecting to Kafka Brokers: {self.kafka_servers}")
        self.consumer = Consumer(self.consumer_conf)
        
        # Cache for loaded FNO models to prevent disk/GPU memory re-allocation thrashing
        self.model_cache: Dict[str, Any] = {}
        
        # Register process interruption signals for graceful exit
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum, frame):
        logger.info("Termination signal received. Gracefully shutting down...")
        self.running = False

    def get_model(self, model_name: str) -> Any:
        """Lazy-load and cache the FNO model state to preserve VRAM."""
        model_name = model_name.lower()
        if model_name not in self.model_cache:
            logger.info(f"Lazy-loading FNO surrogate weights for '{model_name}' on device '{self.device}'...")
            from deepvol.calibration.interface import _get_default_model
            model = _get_default_model(model_name, torch.device(self.device))
            self.model_cache[model_name] = model
        return self.model_cache[model_name]

    def run(self):
        """Main event loop subscribing to options/quotes Kafka topics."""
        logger.info(f"Subscribing to topic: {self.kafka_topic}")
        self.consumer.subscribe([self.kafka_topic])
        
        while self.running:
            try:
                # Poll Kafka broker with a 1.0 second timeout
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())
                
                # Process quote message
                payload = json.loads(msg.value().decode("utf-8"))
                self.process_payload(payload)
                
                # Commit offset manually if auto commit is disabled (At-Least-Once guarantee)
                if not self.consumer_conf["enable.auto.commit"]:
                    self.consumer.commit(msg, asynchronous=True)
                    
            except Exception as e:
                logger.error(f"Error during message consumption/calibration: {e}", exc_info=True)
                # Sleep briefly on error to avoid loop spinning
                time.sleep(1.0)
                
        # Close consumer resources on exit
        logger.info("Closing Kafka consumer...")
        self.consumer.close()

    def process_payload(self, payload: Dict[str, Any]):
        """Parse the payload, run the FNO calibration, and update Redis cache."""
        currency = payload.get("currency", "BTC").upper()
        model_name = payload.get("model_name", "rough_heston").lower()
        timestamp = payload.get("timestamp", time.time())
        
        # 1. Resolve target IV surface from payload
        if "market_iv" in payload:
            # Direct 8x11 IV grid provided
            iv_grid = np.array(payload["market_iv"], dtype=np.float32)
        elif "quotes" in payload:
            # Raw option quotes provided; construct pandas DataFrame and interpolate
            import pandas as pd
            df_quotes = pd.DataFrame(payload["quotes"])
            iv_grid = build_iv_surface(df_quotes, currency=currency)
        else:
            logger.warning("Message payload is missing 'market_iv' and 'quotes'. Skipping.")
            return
            
        if iv_grid.shape != (8, 11):
            logger.error(f"Invalid IV surface shape: {iv_grid.shape}. Expected (8, 11). Skipping.")
            return

        # 2. Retrieve cached model
        model = self.get_model(model_name)
        
        # 3. Execute FNO Calibration (Newton or L-BFGS method)
        logger.info(f"Starting FNO calibration for {currency} using model '{model_name}'...")
        t_start = time.time()
        
        calib_res: CalibrationResult = calibrate(
            market_iv_surface=iv_grid,
            model_name=model_name,
            method="newton",
            device=self.device,
            model=model
        )
        
        elapsed_ms = (time.time() - t_start) * 1000.0
        logger.info(f"Completed {model_name} calibration in {elapsed_ms:.2f}ms. RMSE: {calib_res.rmse:.6f}")
        
        # 4. Serialize parameters (converting numpy array/tensor to list)
        params_dict = {}
        if isinstance(calib_res.parameters, (np.ndarray, torch.Tensor)):
            params_raw = calib_res.parameters
            if isinstance(params_raw, torch.Tensor):
                params_raw = params_raw.cpu().numpy()
            
            # Format according to model parameter names
            if model_name == "heston":
                # [kappa, theta, sigma, rho, v0]
                params_dict = {
                    "kappa": float(params_raw[0]),
                    "theta": float(params_raw[1]),
                    "sigma": float(params_raw[2]),
                    "rho": float(params_raw[3]),
                    "v0": float(params_raw[4]),
                }
            elif model_name == "sabr":
                # [alpha, rho, nu]
                params_dict = {
                    "alpha": float(params_raw[0]),
                    "rho": float(params_raw[1]),
                    "nu": float(params_raw[2]),
                }
            elif model_name in ("rbergomi", "rough_bergomi"):
                # [v0, H, eta, rho]
                params_dict = {
                    "v0": float(params_raw[0]),
                    "H": float(params_raw[1]),
                    "eta": float(params_raw[2]),
                    "rho": float(params_raw[3]),
                }
            else:
                params_dict = {f"param_{i}": float(v) for i, v in enumerate(params_raw.tolist())}
        else:
            params_dict = calib_res.parameters
            
        # 5. Build Redis Cache JSON Payload
        redis_payload = {
            "currency": currency,
            "model_name": model_name,
            "timestamp": timestamp,
            "parameters": params_dict,
            "rmse": calib_res.rmse,
            "rmse_bps": calib_res.rmse * 10000.0,
            "elapsed_ms": elapsed_ms,
            "status": calib_res.status,
            "info": {k: str(v) for k, v in calib_res.info.items()}
        }
        redis_json = json.dumps(redis_payload)
        
        # Update Redis Cache
        latest_key = f"deepvol:calibration:latest:{currency.lower()}:{model_name}"
        history_key = f"deepvol:calibration:history:{currency.lower()}:{model_name}"
        
        pipe = self.redis_client.pipeline()
        # Set latest snapshot
        pipe.set(latest_key, redis_json)
        # Push to historical list (supports dashboard timelines)
        pipe.lpush(history_key, redis_json)
        # Trim history to keep last 1000 points to prevent memory leaks
        pipe.ltrim(history_key, 0, 999)
        pipe.execute()
        
        logger.info(f"Updated Redis key '{latest_key}' and pushed to history list.")


if __name__ == "__main__":
    worker = CalibrationWorker()
    worker.run()
