from dataclasses import dataclass
import time
from pathlib import Path

@dataclass
class MoaiEdgeConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    max_seq_len: int = 64

class DemoRecorderStandalone:
    def __init__(self, config):
        self.config = config
        self.recording = False
        self.steps = []
        
    def start_recording(self):
        self.recording = True
        self.steps = []
        
    def add_human_state(self, human_state, action):
        if self.recording:
            self.steps.append((human_state, action))
            
    def stop_recording(self):
        self.recording = False
        return Path("demo_episode.npz")

class MoaiEdgeClientStandalone:
    def __init__(self, config):
        self.config = config
