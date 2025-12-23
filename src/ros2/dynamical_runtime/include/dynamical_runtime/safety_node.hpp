/**
 * @file safety_node.hpp
 * @brief Safety Shield Node - 1kHz real-time safety monitoring
 *
 * This node runs at 1kHz with SCHED_FIFO priority to ensure
 * safety constraints are NEVER violated. It can override any
 * commanded action with a safe stop.
 *
 * Based on industrial safety standards (ISO 10218, ISO 15066).
 */

#ifndef DYNAMICAL_RUNTIME__SAFETY_NODE_HPP_
#define DYNAMICAL_RUNTIME__SAFETY_NODE_HPP_

#include <memory>
#include <vector>
#include <atomic>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "realtime_tools/realtime_publisher.h"
#include "realtime_tools/realtime_buffer.h"

#include "std_msgs/msg/bool.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "dynamical_msgs/msg/robot_state.hpp"
#include "dynamical_msgs/msg/safety_status.hpp"
#include "dynamical_msgs/msg/safety_violation.hpp"
#include "dynamical_msgs/srv/trigger_estop.hpp"
#include "dynamical_msgs/srv/reset_estop.hpp"

namespace dynamical
{

/**
 * @brief Safety configuration parameters
 */
struct SafetyConfig
{
  // Joint limits margin (radians)
  double joint_limit_margin = 0.087;  // 5 degrees

  // Velocity limit margin (percent)
  double velocity_limit_margin = 0.10;

  // Force/torque limit margin (percent)
  double torque_limit_margin = 0.20;

  // Obstacle distances (meters)
  double min_obstacle_distance = 0.10;
  double human_safety_distance = 0.50;

  // Watchdog timeout (milliseconds)
  int watchdog_timeout_ms = 10;

  // E-stop deceleration (rad/s^2)
  double estop_deceleration = 10.0;
};

/**
 * @brief Safety Shield Node
 *
 * Runs at 1kHz to monitor all safety constraints.
 * Uses lifecycle for controlled startup/shutdown.
 */
class SafetyNode : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit SafetyNode(const rclcpp::NodeOptions & options);
  ~SafetyNode() override;

  // Lifecycle callbacks
  CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;

private:
  // Main safety loop (1kHz)
  void safety_loop();

  // Safety checks
  bool check_joint_limits();
  bool check_velocity_limits();
  bool check_torque_limits();
  bool check_obstacle_proximity();
  bool check_human_proximity();
  bool check_watchdog();

  // E-stop handling
  void trigger_estop(const std::string & reason);
  bool reset_estop();

  // Callbacks
  void robot_state_callback(const dynamical_msgs::msg::RobotState::SharedPtr msg);
  void obstacles_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg);
  void humans_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg);
  void heartbeat_callback(const std_msgs::msg::Bool::SharedPtr msg);

  // Service callbacks
  void handle_trigger_estop(
    const std::shared_ptr<dynamical_msgs::srv::TriggerEstop::Request> request,
    std::shared_ptr<dynamical_msgs::srv::TriggerEstop::Response> response);
  void handle_reset_estop(
    const std::shared_ptr<dynamical_msgs::srv::ResetEstop::Request> request,
    std::shared_ptr<dynamical_msgs::srv::ResetEstop::Response> response);

  // Real-time safe data access
  realtime_tools::RealtimeBuffer<dynamical_msgs::msg::RobotState> robot_state_buffer_;
  realtime_tools::RealtimeBuffer<std::vector<geometry_msgs::msg::Pose>> obstacles_buffer_;
  realtime_tools::RealtimeBuffer<std::vector<geometry_msgs::msg::Pose>> humans_buffer_;

  // Publishers (real-time safe)
  std::shared_ptr<realtime_tools::RealtimePublisher<dynamical_msgs::msg::SafetyStatus>>
    safety_status_pub_;
  std::shared_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Bool>>
    estop_pub_;

  // Subscribers
  rclcpp::Subscription<dynamical_msgs::msg::RobotState>::SharedPtr robot_state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr obstacles_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr humans_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr heartbeat_sub_;

  // Services
  rclcpp::Service<dynamical_msgs::srv::TriggerEstop>::SharedPtr trigger_estop_srv_;
  rclcpp::Service<dynamical_msgs::srv::ResetEstop>::SharedPtr reset_estop_srv_;

  // Timer for main loop
  rclcpp::TimerBase::SharedPtr timer_;

  // Configuration
  SafetyConfig config_;
  std::vector<double> joint_limits_lower_;
  std::vector<double> joint_limits_upper_;
  std::vector<double> velocity_limits_;
  std::vector<double> torque_limits_;
  size_t num_joints_;

  // State
  std::atomic<bool> estop_active_{false};
  std::atomic<uint64_t> last_heartbeat_time_{0};
  std::atomic<uint32_t> missed_deadlines_{0};

  // Current violations
  std::mutex violations_mutex_;
  std::vector<dynamical_msgs::msg::SafetyViolation> current_violations_;

  // Timing
  std::chrono::steady_clock::time_point last_loop_time_;
  double loop_time_us_ = 0.0;
};

}  // namespace dynamical

#endif  // DYNAMICAL_RUNTIME__SAFETY_NODE_HPP_
