"""
Neuracore Client for Asynchronous Data Streaming.

This module implements the asynchronous data logging pattern described in the Neuracore report.
It decouples sensor reading from disk/network writing using a background thread and queue.
"""

import threading
import queue
import time
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

class NeuracoreClient:
    """
    Client for asynchronous data streaming to Neuracore Cloud or local storage.
    
    Features:
    - Asynchronous logging via background thread
    - Ring buffer/Queue for decoupling
    - Schema validation (placeholder)
    """
    
    def __init__(
        self,
        api_url: str = "https://api.neuracore.app/api",
        org_id: Optional[str] = None,
        storage_path: Optional[str] = None,
        queue_size: int = 1000,
        flush_interval: float = 1.0
    ):
        self.api_url = api_url
        self.org_id = org_id or os.environ.get("NEURACORE_ORG_ID")
        self.storage_path = Path(storage_path) if storage_path else None
        
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.worker_thread = None
        self.flush_interval = flush_interval
        
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
    def start(self):
        """Start the background logging thread."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Neuracore async client started")
        
    def stop(self):
        """Stop the background thread and flush remaining data."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("Neuracore async client stopped")
        
    def log(self, channel: str, data: Any, timestamp: float = None):
        """
        Log data to a specific channel asynchronously.
        
        Args:
            channel: Data channel (e.g., 'camera_0', 'joint_states')
            data: Data payload (dict, array, etc.)
            timestamp: Timestamp in seconds (default: current time)
        """
        if not self.running:
            logger.warning("Neuracore client not running, dropping data")
            return
            
        if timestamp is None:
            timestamp = time.time()
            
        try:
            self.queue.put_nowait({
                'channel': channel,
                'data': data,
                'timestamp': timestamp
            })
        except queue.Full:
            logger.warning("Neuracore logging queue full, dropping frame")
            
    def _worker_loop(self):
        """Background worker to process the queue."""
        buffer = []
        last_flush = time.time()
        
        while self.running or not self.queue.empty():
            try:
                # Wait for data with timeout to allow periodic flushing
                item = self.queue.get(timeout=0.1)
                buffer.append(item)
            except queue.Empty:
                pass
                
            # Flush if buffer full or interval elapsed
            current_time = time.time()
            if len(buffer) >= 100 or (current_time - last_flush > self.flush_interval and buffer):
                self._flush(buffer)
                buffer = []
                last_flush = current_time
                
    def _flush(self, buffer: list):
        """Write buffer to storage/network."""
        # For this implementation, we'll write to local JSONL files per channel
        # In a real implementation, this would POST to the API
        
        by_channel = {}
        for item in buffer:
            ch = item['channel']
            if ch not in by_channel:
                by_channel[ch] = []
            by_channel[ch].append(item)
            
        if self.storage_path:
            for ch, items in by_channel.items():
                file_path = self.storage_path / f"{ch}.jsonl"
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'a') as f:
                    for item in items:
                        # Handle numpy serialization
                        json_str = json.dumps(item, default=self._json_serializer)
                        f.write(json_str + '\n')
                        
    @staticmethod
    def _json_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

# Singleton instance
_client = None

def get_client(**kwargs) -> NeuracoreClient:
    global _client
    if _client is None:
        _client = NeuracoreClient(**kwargs)
    return _client
