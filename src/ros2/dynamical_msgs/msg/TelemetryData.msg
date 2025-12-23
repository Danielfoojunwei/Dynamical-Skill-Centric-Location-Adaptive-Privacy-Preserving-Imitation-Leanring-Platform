# Telemetry data for logging and analytics

std_msgs/Header header

# Timing
float64 tier1_loop_time_us
float64 tier2_loop_time_us
float64 policy_inference_time_us
uint32 tier1_missed_deadlines

# Perception
uint8 perception_cascade_level
float64 perception_latency_ms

# Safety
uint8 safety_status
uint32 safety_violations_count

# Skill execution
string current_skill_id
float32 skill_progress

# Resource metrics
float32 cpu_temp_celsius
float32 gpu_temp_celsius
float32 power_watts

# Network
bool cloud_connected
float32 upload_queue_size
