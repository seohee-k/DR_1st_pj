#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <Eigen/Dense>
#include <cmath>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_interface/planning_interface.h>


#define DEG2RAD(x) ((x) * M_PI / 180.0)

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  
  auto const node = std::make_shared<rclcpp::Node>(
    "init_pose",
    rclcpp::NodeOptions()
      .automatically_declare_parameters_from_overrides(true)
      .append_parameter_override("use_sim_time", true)
  );

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  auto const logger = rclcpp::get_logger("init_pose");

  moveit::planning_interface::MoveGroupInterface move_group(node, "arm");

  rclcpp::sleep_for(std::chrono::seconds(2));

  // 원하는 목표 joint 각도 설정 (라디안 단위)
  // 예시: theta1, theta2, theta3, theta4 = 0.0, -0.5, 0.5, 0.0
  std::vector<double> joint_positions = {DEG2RAD(0), DEG2RAD(-73), DEG2RAD(78), DEG2RAD(11)};

  move_group.setJointValueTarget(joint_positions);

  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

  if (success) {
    move_group.execute(my_plan);
    RCLCPP_INFO(logger, "Moved to the target joint positions.");
  } else {
    RCLCPP_ERROR(logger, "Failed to plan to the target joint positions.");
  }

  rclcpp::shutdown();
  spinner.join();
  return 0;
}
