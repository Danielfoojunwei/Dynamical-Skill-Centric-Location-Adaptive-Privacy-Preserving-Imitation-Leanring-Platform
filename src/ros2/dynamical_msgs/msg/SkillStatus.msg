# Skill execution status

std_msgs/Header header

# Skill identification
string skill_id
string skill_name

# Execution states
uint8 STATE_IDLE = 0
uint8 STATE_LOADING = 1
uint8 STATE_RUNNING = 2
uint8 STATE_PAUSED = 3
uint8 STATE_COMPLETING = 4
uint8 STATE_COMPLETED = 5
uint8 STATE_FAILED = 6
uint8 STATE_CANCELLED = 7

uint8 state

# Progress
float32 progress              # 0.0 to 1.0
float64 elapsed_time_sec
float64 estimated_remaining_sec

# Error info (if failed)
string error_message
uint32 error_code

# Skill parameters
string parameters_json
