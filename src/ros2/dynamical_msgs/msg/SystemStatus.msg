# Overall system status

std_msgs/Header header

# System modes
uint8 MODE_STOPPED = 0
uint8 MODE_INITIALIZING = 1
uint8 MODE_READY = 2
uint8 MODE_RUNNING = 3
uint8 MODE_PAUSED = 4
uint8 MODE_ERROR = 5
uint8 MODE_OFFLINE = 6

uint8 mode

# Component status
bool safety_node_active
bool perception_node_active
bool policy_node_active
bool actuator_node_active

# Timing metrics (Hz)
float32 safety_loop_rate
float32 perception_loop_rate
float32 policy_loop_rate
float32 control_loop_rate

# Resource usage
float32 cpu_usage_percent
float32 gpu_usage_percent
float32 memory_usage_percent
float32 gpu_memory_usage_mb

# Network status
bool cloud_connected
float32 cloud_latency_ms

# Skills
uint32 cached_skills_count
string[] active_skill_ids
