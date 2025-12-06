"""
GMR (Giant Magnetoresistive) Sensor Driver
for Dynamical Edge Platform

This module handles communication with GMR sensors used for precise
magnetic field sensing (e.g., for non-contact position tracking or
material detection).

Includes a simulator for development without hardware.
"""

import time
import threading
import queue
import json
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class GMRReading:
    """A single reading from the GMR sensor."""
    timestamp: float
    x: float  # Magnetic field X (Gauss or Tesla)
    y: float  # Magnetic field Y
    z: float  # Magnetic field Z
    temperature: float = 25.0  # Sensor temperature (C)

class GMRDriverBase(ABC):
    """Abstract base class for GMR drivers."""
    
    @abstractmethod
    def connect(self) -> bool:
        pass
        
    @abstractmethod
    def disconnect(self):
        pass
        
    @abstractmethod
    def read(self) -> Optional[GMRReading]:
        pass
        
    @abstractmethod
    def zero(self) -> bool:
        """Tare/Zero the sensor at current field."""
        pass

class GMRDriver(GMRDriverBase):
    """
    Serial driver for generic GMR sensors (ASCII protocol).
    Protocol: "X:12.3 Y:45.6 Z:78.9 T:25.0\\n"
    """
    
    def __init__(self, port: str = "/dev/ttyUSB1", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._serial = None
        self._connected = False
        self._offsets = np.zeros(3)
        
    def connect(self) -> bool:
        try:
            import serial
            self._serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self._connected = True
            logger.info(f"GMR Sensor connected on {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect GMR: {e}")
            return False
            
    def disconnect(self):
        if self._serial:
            self._serial.close()
            self._serial = None
        self._connected = False
        
    def read(self) -> Optional[GMRReading]:
        if not self._connected or not self._serial:
            return None
            
        try:
            # Read line
            line = self._serial.readline().decode('utf-8').strip()
            if not line:
                return None
                
            # Parse simple key-value format
            parts = line.split()
            data = {}
            for part in parts:
                if ':' in part:
                    k, v = part.split(':')
                    data[k] = float(v)
            
            raw = np.array([data.get('X', 0), data.get('Y', 0), data.get('Z', 0)])
            calibrated = raw - self._offsets
            
            return GMRReading(
                timestamp=time.time(),
                x=calibrated[0],
                y=calibrated[1],
                z=calibrated[2],
                temperature=data.get('T', 25.0)
            )
            
        except Exception as e:
            logger.error(f"GMR Read Error: {e}")
            return None

    def zero(self) -> bool:
        """Read 10 samples and set as offset."""
        if not self._connected:
            return False
            
        samples = []
        for _ in range(10):
            reading = self.read()
            if reading:
                # Add back offset to get raw
                raw = np.array([reading.x, reading.y, reading.z]) + self._offsets
                samples.append(raw)
            time.sleep(0.05)
            
        if samples:
            self._offsets = np.mean(samples, axis=0)
            logger.info(f"GMR Zeroed. Offsets: {self._offsets}")
            return True
        return False

class GMRSimulator(GMRDriverBase):
    """Simulates a GMR sensor with synthetic magnetic fields."""
    
    def __init__(self):
        self._connected = False
        self._offsets = np.zeros(3)
        self._start_time = time.time()
        
    def connect(self) -> bool:
        self._connected = True
        logger.info("GMR Simulator connected")
        return True
        
    def disconnect(self):
        self._connected = False
        
    def read(self) -> Optional[GMRReading]:
        if not self._connected:
            return None
            
        t = time.time() - self._start_time
        
        # Simulate a rotating magnetic field + noise
        raw_x = 50.0 * np.sin(t) + np.random.normal(0, 0.5)
        raw_y = 50.0 * np.cos(t) + np.random.normal(0, 0.5)
        raw_z = 10.0 * np.sin(t * 0.5) + np.random.normal(0, 0.5)
        
        raw = np.array([raw_x, raw_y, raw_z])
        calibrated = raw - self._offsets
        
        return GMRReading(
            timestamp=time.time(),
            x=calibrated[0],
            y=calibrated[1],
            z=calibrated[2],
            temperature=35.0 + np.sin(t*0.1)
        )
        
    def zero(self) -> bool:
        # In sim, just set offsets to current "raw" value
        t = time.time() - self._start_time
        self._offsets = np.array([
            50.0 * np.sin(t),
            50.0 * np.cos(t),
            10.0 * np.sin(t * 0.5)
        ])
        logger.info(f"GMR Simulator Zeroed. Offsets: {self._offsets}")
        return True
