# Object detection result

string class_name
uint32 class_id
float32 confidence

# 2D bounding box (image coordinates)
float32 bbox_x
float32 bbox_y
float32 bbox_width
float32 bbox_height

# 3D position (robot frame)
geometry_msgs/Point position
float32 distance

# Velocity (if tracked)
geometry_msgs/Vector3 velocity
bool is_tracked
uint32 track_id

# Flags
bool is_obstacle
bool is_human
bool in_workspace
