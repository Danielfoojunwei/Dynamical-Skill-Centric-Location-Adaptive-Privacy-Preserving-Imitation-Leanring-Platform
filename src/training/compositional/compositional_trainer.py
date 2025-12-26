"""
Compositional Trainer - End-to-End Compositional Training Pipeline

Integrates:
1. Data Capture (ONVIF cameras + MANUS gloves + robot proprio)
2. Skill Discovery (SAM3 + DINOv3 + V-JEPA2 segmentation)
3. Hierarchy Learning (temporal structure from V-JEPA2)
4. Primitive Training (individual skill policies)
5. Composition Training (sequence/conditional operators)

This enables:
- Training compositional policies from continuous human demos
- Novel task composition at inference time
- Data-efficient learning (O(n) primitives vs O(n²) combinations)

Integration with existing infrastructure:
- src/platform/edge/onvif_cameras.py - video capture
- src/platform/edge/manus_sdk.py - glove data
- src/drivers/dyglove.py - alternative glove interface
- src/meta_ai/ - SAM3, DINOv3, V-JEPA2
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Iterator

import numpy as np

from .skill_discovery import (
    SkillDiscovery,
    DiscoveryConfig,
    DiscoveredPrimitive,
    DemoSegment,
)
from .hierarchical_imitation import (
    HierarchicalImitation,
    HierarchyConfig,
    SkillHierarchy,
    SkillNode,
)

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Current phase of compositional training."""
    DATA_COLLECTION = "data_collection"
    SKILL_DISCOVERY = "skill_discovery"
    HIERARCHY_LEARNING = "hierarchy_learning"
    PRIMITIVE_TRAINING = "primitive_training"
    COMPOSITION_TRAINING = "composition_training"
    EVALUATION = "evaluation"


@dataclass
class CompositionalConfig:
    """Configuration for compositional training."""
    # Data capture
    onvif_camera_urls: List[str] = field(default_factory=list)
    use_manus_gloves: bool = True
    capture_fps: int = 30

    # Skill discovery
    discovery_config: Optional[DiscoveryConfig] = None

    # Hierarchy learning
    hierarchy_config: Optional[HierarchyConfig] = None

    # Training
    primitive_epochs: int = 100
    composition_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4

    # Output
    output_dir: str = "checkpoints/compositional"
    save_every_n_demos: int = 100

    # Hardware
    device: str = "cuda"

    # Continuous learning
    incremental_discovery: bool = True
    discovery_interval: int = 100  # Demos between discovery runs

    @classmethod
    def for_jetson_thor(cls) -> "CompositionalConfig":
        """Config optimized for Jetson Thor."""
        return cls(
            device="cuda",
            batch_size=16,  # Lower for memory
            capture_fps=30,
        )


@dataclass
class DemoCapture:
    """A captured demonstration."""
    demo_id: str
    timestamp: float

    # Visual data
    frames: List[np.ndarray] = field(default_factory=list)

    # Action data
    actions: List[np.ndarray] = field(default_factory=list)
    proprio: List[np.ndarray] = field(default_factory=list)

    # Glove data
    hand_pose: List[np.ndarray] = field(default_factory=list)
    finger_pressures: List[np.ndarray] = field(default_factory=list)

    # Timestamps
    frame_timestamps: List[float] = field(default_factory=list)

    # Metadata
    task_instruction: Optional[str] = None


@dataclass
class TrainingResult:
    """Result from compositional training."""
    # Discovered primitives
    num_primitives: int = 0
    primitives: List[DiscoveredPrimitive] = field(default_factory=list)

    # Hierarchy
    hierarchy_stats: Dict[str, Any] = field(default_factory=dict)

    # Training metrics
    primitive_losses: Dict[str, List[float]] = field(default_factory=dict)
    composition_loss: List[float] = field(default_factory=list)

    # Checkpoints
    checkpoint_paths: List[str] = field(default_factory=list)

    # Timing
    total_training_time: float = 0.0
    phase_times: Dict[str, float] = field(default_factory=dict)


class DataCaptureStream:
    """
    Stream of demonstration data from ONVIF cameras and MANUS gloves.

    Integrates with existing platform infrastructure:
    - src/platform/edge/onvif_cameras.py
    - src/platform/edge/manus_sdk.py
    """

    def __init__(self, config: CompositionalConfig):
        self.config = config

        # Camera interface
        self._cameras = None

        # Glove interface
        self._gloves = None

        # Current capture
        self._current_demo: Optional[DemoCapture] = None
        self._demo_count = 0

    def initialize(self):
        """Initialize capture devices."""
        # Initialize ONVIF cameras
        if self.config.onvif_camera_urls:
            try:
                from ...platform.edge.onvif_cameras import ONVIFCameraManager
                self._cameras = ONVIFCameraManager(self.config.onvif_camera_urls)
                self._cameras.connect()
                logger.info(f"Connected to {len(self.config.onvif_camera_urls)} ONVIF cameras")
            except ImportError:
                logger.warning("ONVIF camera module not available")

        # Initialize MANUS gloves
        if self.config.use_manus_gloves:
            try:
                from ...platform.edge.manus_sdk import ManusGloveInterface
                self._gloves = ManusGloveInterface()
                self._gloves.connect()
                logger.info("Connected to MANUS gloves")
            except ImportError:
                try:
                    from ...drivers.dyglove import DyGloveInterface
                    self._gloves = DyGloveInterface()
                    self._gloves.connect()
                    logger.info("Connected to DyGlove (fallback)")
                except ImportError:
                    logger.warning("No glove interface available")

    def start_demo(self, task_instruction: Optional[str] = None) -> str:
        """Start capturing a new demonstration."""
        self._demo_count += 1
        demo_id = f"demo_{self._demo_count}_{int(time.time())}"

        self._current_demo = DemoCapture(
            demo_id=demo_id,
            timestamp=time.time(),
            task_instruction=task_instruction,
        )

        logger.info(f"Started demo capture: {demo_id}")
        return demo_id

    def capture_frame(
        self,
        robot_action: Optional[np.ndarray] = None,
        robot_proprio: Optional[np.ndarray] = None,
    ):
        """Capture a single frame of demonstration data."""
        if self._current_demo is None:
            return

        timestamp = time.time()
        self._current_demo.frame_timestamps.append(timestamp)

        # Capture camera frames
        if self._cameras is not None:
            frames = self._cameras.capture_all()
            # Stack multi-camera frames
            self._current_demo.frames.append(np.stack(frames))
        else:
            # Placeholder for testing
            self._current_demo.frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        # Capture glove data
        if self._gloves is not None:
            hand_data = self._gloves.get_hand_pose()
            self._current_demo.hand_pose.append(hand_data['pose'])
            if 'pressure' in hand_data:
                self._current_demo.finger_pressures.append(hand_data['pressure'])

        # Store robot data
        if robot_action is not None:
            self._current_demo.actions.append(robot_action)
        if robot_proprio is not None:
            self._current_demo.proprio.append(robot_proprio)

    def end_demo(self) -> DemoCapture:
        """End current demonstration and return captured data."""
        demo = self._current_demo
        self._current_demo = None

        logger.info(f"Ended demo capture: {demo.demo_id}, {len(demo.frames)} frames")
        return demo

    def stream_demos(
        self,
        demo_callback: Optional[Callable[[DemoCapture], None]] = None,
    ) -> Iterator[DemoCapture]:
        """
        Continuously stream demonstrations.

        This integrates with continuous human trainer capture.
        """
        logger.info("Starting continuous demo stream...")

        while True:
            # Wait for demo start signal (e.g., button press, voice command)
            # For now, yield when demo is complete
            if self._current_demo is not None and len(self._current_demo.frames) > 0:
                demo = self.end_demo()
                if demo_callback:
                    demo_callback(demo)
                yield demo

            time.sleep(0.1)


class CompositionalTrainer:
    """
    End-to-end compositional training pipeline.

    Brings together:
    1. Data capture from ONVIF/MANUS
    2. Skill discovery using Meta AI
    3. Hierarchy learning
    4. Primitive and composition training
    """

    def __init__(self, config: Optional[CompositionalConfig] = None):
        self.config = config or CompositionalConfig()

        # Data capture
        self.data_stream = DataCaptureStream(self.config)

        # Skill discovery
        discovery_config = self.config.discovery_config or DiscoveryConfig(
            device=self.config.device
        )
        self.skill_discovery = SkillDiscovery(discovery_config)

        # Hierarchy learning (initialized after discovery)
        self.hierarchy: Optional[HierarchicalImitation] = None

        # State
        self.phase = TrainingPhase.DATA_COLLECTION
        self.demos: List[DemoCapture] = []
        self.primitives: List[DiscoveredPrimitive] = []

        # Training state
        self._primitive_policies: Dict[str, Any] = {}
        self._composition_model: Optional[Any] = None

    def initialize(self):
        """Initialize all components."""
        logger.info("Initializing compositional trainer...")

        # Initialize data capture
        self.data_stream.initialize()

        # Load Meta AI models for discovery
        self.skill_discovery.segmenter.load_models()

        logger.info("Compositional trainer ready")

    def add_demo(self, demo: DemoCapture):
        """Add a demonstration to the training set."""
        self.demos.append(demo)
        logger.info(f"Added demo {demo.demo_id}, total: {len(self.demos)}")

        # Incremental discovery
        if (self.config.incremental_discovery and
            len(self.demos) % self.config.discovery_interval == 0):
            logger.info("Running incremental skill discovery...")
            self._run_discovery()

    def _run_discovery(self):
        """Run skill discovery on current demos."""
        self.phase = TrainingPhase.SKILL_DISCOVERY

        # Convert demos to discovery format
        demo_dicts = []
        for demo in self.demos:
            demo_dicts.append({
                'frames': demo.frames,
                'actions': demo.actions,
                'proprio': demo.proprio,
                'hand_pose': demo.hand_pose,
                'timestamps': demo.frame_timestamps,
            })

        # Discover primitives
        self.primitives = self.skill_discovery.discover_from_demos(demo_dicts)

        logger.info(f"Discovered {len(self.primitives)} primitives")

    def learn_hierarchy(self):
        """Learn skill hierarchy from discovered primitives."""
        self.phase = TrainingPhase.HIERARCHY_LEARNING

        if not self.primitives:
            logger.warning("No primitives discovered, running discovery first")
            self._run_discovery()

        # Initialize hierarchy learning
        hierarchy_config = self.config.hierarchy_config or HierarchyConfig(
            device=self.config.device
        )
        self.hierarchy = HierarchicalImitation(self.primitives, hierarchy_config)

        # Extract primitive sequences from demos
        primitive_sequences = self._extract_primitive_sequences()

        # Learn composition
        self.hierarchy.learn_composition(primitive_sequences)

        stats = self.hierarchy.get_hierarchy_stats()
        logger.info(f"Learned hierarchy: {stats}")

    def _extract_primitive_sequences(self) -> List[List[str]]:
        """Extract primitive ID sequences from demos."""
        sequences = []

        for demo in self.demos:
            # Segment demo
            demo_dict = {
                'frames': demo.frames,
                'actions': demo.actions,
                'proprio': demo.proprio,
                'hand_pose': demo.hand_pose,
                'timestamps': demo.frame_timestamps,
            }

            segments, _ = self.skill_discovery.segmenter.segment_demonstration(
                **demo_dict
            )

            # Match segments to primitives
            sequence = []
            for segment in segments:
                # Find closest primitive by features
                best_match = self._match_segment_to_primitive(segment)
                if best_match:
                    sequence.append(best_match)

            if sequence:
                sequences.append(sequence)

        return sequences

    def _match_segment_to_primitive(self, segment: DemoSegment) -> Optional[str]:
        """Match a segment to its closest primitive."""
        if not self.primitives:
            return None

        # Extract features
        if segment.dino_features is not None:
            feat = segment.dino_features.mean(axis=0)
        else:
            # Fallback
            return self.primitives[0].primitive_id

        # Find closest primitive by centroid distance
        best_dist = float('inf')
        best_prim = None

        for prim in self.primitives:
            if prim.centroid_features is not None:
                dist = np.linalg.norm(feat - prim.centroid_features)
                if dist < best_dist:
                    best_dist = dist
                    best_prim = prim.primitive_id

        return best_prim

    def train_primitives(self):
        """Train individual primitive policies."""
        self.phase = TrainingPhase.PRIMITIVE_TRAINING

        if not self.primitives:
            raise RuntimeError("No primitives to train")

        logger.info(f"Training {len(self.primitives)} primitive policies...")

        for prim in self.primitives:
            logger.info(f"Training primitive: {prim.name}")

            # Collect training data from exemplar segments
            train_data = self._prepare_primitive_data(prim)

            # Train policy
            policy = self._train_primitive_policy(prim, train_data)

            # Store
            self._primitive_policies[prim.primitive_id] = policy

            # Update hierarchy
            if self.hierarchy:
                node = self.hierarchy.hierarchy.all_nodes.get(prim.primitive_id)
                if node:
                    node.policy = policy

        logger.info("Primitive training complete")

    def _prepare_primitive_data(self, prim: DiscoveredPrimitive) -> Dict[str, Any]:
        """Prepare training data for a primitive."""
        all_frames = []
        all_actions = []
        all_proprio = []

        for segment in prim.exemplar_segments:
            all_frames.extend(segment.frames)
            all_actions.extend(segment.actions)
            all_proprio.extend(segment.proprio)

        return {
            'frames': np.array(all_frames),
            'actions': np.array(all_actions),
            'proprio': np.array(all_proprio),
        }

    def _train_primitive_policy(
        self,
        prim: DiscoveredPrimitive,
        data: Dict[str, Any],
    ) -> Any:
        """Train a policy for a single primitive."""
        # This would use the existing training infrastructure
        # For now, return a placeholder

        logger.info(f"Training {prim.name} on {len(data['frames'])} frames")

        # Placeholder - would integrate with actual policy training
        class MockPolicy:
            def __init__(self, name):
                self.name = name

            def __call__(self, obs):
                return np.zeros(7)

        return MockPolicy(prim.name)

    def train_composition(self):
        """Train composition operators."""
        self.phase = TrainingPhase.COMPOSITION_TRAINING

        if self.hierarchy is None:
            raise RuntimeError("Hierarchy not learned")

        logger.info("Training composition operators...")

        # Train transition models for sequences
        for skill in self.hierarchy.hierarchy.basic_skills:
            if skill.composition:
                self._train_transition_model(skill)

        logger.info("Composition training complete")

    def _train_transition_model(self, skill: SkillNode):
        """Train transition model for a composed skill."""
        # Extract transition data between children
        # Train model to predict transition dynamics
        pass

    def train(self) -> TrainingResult:
        """Run full training pipeline."""
        start_time = time.time()
        result = TrainingResult()
        phase_times = {}

        # Phase 1: Skill Discovery
        phase_start = time.time()
        self._run_discovery()
        phase_times['skill_discovery'] = time.time() - phase_start
        result.primitives = self.primitives
        result.num_primitives = len(self.primitives)

        # Phase 2: Hierarchy Learning
        phase_start = time.time()
        self.learn_hierarchy()
        phase_times['hierarchy_learning'] = time.time() - phase_start
        if self.hierarchy:
            result.hierarchy_stats = self.hierarchy.get_hierarchy_stats()

        # Phase 3: Primitive Training
        phase_start = time.time()
        self.train_primitives()
        phase_times['primitive_training'] = time.time() - phase_start

        # Phase 4: Composition Training
        phase_start = time.time()
        self.train_composition()
        phase_times['composition_training'] = time.time() - phase_start

        # Save checkpoints
        self._save_checkpoints(result)

        result.total_training_time = time.time() - start_time
        result.phase_times = phase_times

        logger.info(f"Training complete in {result.total_training_time:.1f}s")
        return result

    def _save_checkpoints(self, result: TrainingResult):
        """Save training checkpoints."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save primitives
        for prim in self.primitives:
            path = output_dir / f"primitive_{prim.primitive_id}.pt"
            # torch.save(self._primitive_policies[prim.primitive_id], path)
            result.checkpoint_paths.append(str(path))

        # Save hierarchy
        hierarchy_path = output_dir / "hierarchy.json"
        result.checkpoint_paths.append(str(hierarchy_path))

    def compose_novel_task(self, task_description: str) -> Optional[SkillNode]:
        """
        Compose a skill hierarchy for a novel task.

        This is the key capability enabled by compositional training:
        Handle tasks NOT seen during training!
        """
        if self.hierarchy is None:
            raise RuntimeError("Hierarchy not trained")

        return self.hierarchy.compose_for_task(task_description)

    def execute_composed_task(
        self,
        task_node: SkillNode,
        state: Any,
    ) -> List[Any]:
        """Execute a composed task."""
        def executor_fn(primitive_id: str, state: Any) -> Dict[str, Any]:
            policy = self._primitive_policies.get(primitive_id)
            if policy:
                action = policy(state)
                return {'action': action, 'state': state}
            return {}

        return self.hierarchy.execute_composed_skill(task_node, executor_fn, state)

    @classmethod
    def for_jetson_thor(cls) -> "CompositionalTrainer":
        """Create trainer optimized for Jetson Thor."""
        return cls(CompositionalConfig.for_jetson_thor())
