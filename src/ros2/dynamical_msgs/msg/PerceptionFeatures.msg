# Perception pipeline output
# Published at 30Hz (Level 1) or on-demand (Level 2-3)

std_msgs/Header header

# Cascade level
uint8 CASCADE_LEVEL_1 = 1   # Always (30Hz, <10ms)
uint8 CASCADE_LEVEL_2 = 2   # On-demand (10Hz, <50ms)
uint8 CASCADE_LEVEL_3 = 3   # Rare (1-5Hz, <500ms)

uint8 cascade_level

# Detections (all levels)
Detection[] detections

# Depth (Level 1+)
sensor_msgs/Image depth_map

# Dense features (Level 2+)
float32[] dense_features
uint32 feature_dim

# Segmentation (Level 2+)
sensor_msgs/Image segmentation_mask
string[] class_names

# Scene understanding (Level 3)
string scene_description
float32[] object_affordances

# Timing
float64 inference_time_ms
