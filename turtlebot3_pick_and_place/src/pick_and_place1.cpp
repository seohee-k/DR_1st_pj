#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_interface/planning_interface.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include "aruco_msgs/msg/marker_array.hpp"

#include <Eigen/Dense>
#include <cmath>

#define DEG2RAD(x) ((x) * M_PI / 180.0)
#define RAD2DEG(x) ((x) * 180.0 / M_PI)

bool marker_saved = false;
double x_ob = 0.0, y_ob = 0.0, z_ob = 0.0;

Eigen::Matrix4d std_DH(double theta, double alpha, double a, double d) {
    Eigen::Matrix4d DH;
    DH << cos(theta), -sin(theta) * cos(alpha), sin(theta)* sin(alpha), a* cos(theta),
        sin(theta), cos(theta)* cos(alpha), -cos(theta) * sin(alpha), a* sin(theta),
        0, sin(alpha), cos(alpha), d,
        0, 0, 0, 1;
    return DH;
}

Eigen::Matrix4d forward_kinematics(double q1, double q2, double q3, double q4) {
    Eigen::Matrix4d T01 = std_DH(q1, M_PI / 2, 0.0, 0.077);
    Eigen::Matrix4d T12 = std_DH(q2 + M_PI / 2, 0.0, 0.128, 0.0);
    Eigen::Matrix4d T23 = std_DH(q3 - M_PI / 2, 0.0, 0.148, 0.0);
    Eigen::Matrix4d T34 = std_DH(q4, 0.0, 0.126, 0.0);

    Eigen::Matrix4d T04 = T01 * T12 * T23 * T34;
    return T04;
}

struct Pose {
    double px, py, pz;  // End-effector position
    double phi;         // Orientation (theta2 + theta3 + theta4)
};

struct IKResult {
    double theta1, theta2, theta3, theta4;
    bool valid;
};

IKResult solveIK(const Pose& pose, double a2, double a3, double a4, double d1, bool elbow_up = true) {
    IKResult result;
    result.valid = true;

    // Step 1: theta1
    result.theta1 = atan2(pose.py, pose.px);

    // Step 2: r3, z3
    double pr = sqrt(pose.px * pose.px + pose.py * pose.py);
    double r3 = pr;
    double z3 = pose.pz - d1;

    // Step 3: r2, z2 (with phi input)
    double r2 = r3 - a4 * cos(pose.phi);
    double z2 = z3 - a4 * sin(pose.phi);

    // Step 4: theta3
    double D = (r2 * r2 + z2 * z2 - a2 * a2 - a3 * a3) / (2 * a2 * a3);
    if (D < -1.0 || D > 1.0) {
        std::cerr << "[IK] No solution: acos out of range.\n";
        result.valid = false;
        return result;
    }

    double theta3 = acos(D);
    if (!elbow_up) theta3 = -theta3; // elbow-down solution
    double raw_theta3 = theta3;
    result.theta3 = -(raw_theta3 + M_PI / 2);
    
    //step 5:theta2
    double raw_theta2 = atan2(z2, r2) - atan2(a3 * sin(theta3), a2 + a3 * cos(theta3));
    result.theta2 = -(raw_theta2 - M_PI / 2);
    
    
    // Step 6: theta4
    double raw_theta4 = pose.phi - (raw_theta2 + raw_theta3);
    result.theta4 = -raw_theta4;
    return result;
}


Eigen::Vector3d convertToBaseFrame(const Eigen::Vector3d& tvec_optical) {
    // z축 +90도 회전 행렬 (Rz)
    double theta_z = -M_PI / 2.0;
    Eigen::Matrix3d Rz;
    Rz << cos(theta_z), -sin(theta_z), 0,
          sin(theta_z),  cos(theta_z), 0,
          0,             0,            1;
          
    // x축 +90도 회전 행렬 (Rx)
    double theta_x = -M_PI / 2.0;
    Eigen::Matrix3d Rx;
    Rx << 1, 0,             0,
          0, cos(theta_x), -sin(theta_x),
          0, sin(theta_x),  cos(theta_x);
          
     // optical → camera_rgb_frame 변환 행렬
    Eigen::Matrix3d R_optical_to_rgb = Rz * Rx;
     // optical frame에서의 위치를 camera_rgb_frame 기준으로 변환
    Eigen::Vector3d tvec_in_eeframe = R_optical_to_rgb * tvec_optical;

    return tvec_in_eeframe;
}


void markerCallback(const aruco_msgs::msg::MarkerArray::SharedPtr msg) {
    if (marker_saved) return;
    
    for (const auto& marker : msg->markers) {
        if (marker.id == 1) {  // 원하는 ID
            x_ob = marker.pose.pose.position.x;
            y_ob = marker.pose.pose.position.y;
            z_ob = marker.pose.pose.position.z;
            marker_saved = true;
            break;
        }
    }
}


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    auto const node = std::make_shared<rclcpp::Node>(
      "pick_and_place",
      rclcpp::NodeOptions()
        .automatically_declare_parameters_from_overrides(true)
        .append_parameter_override("use_sim_time", true)
    );

    auto marker_sub = node->create_subscription<aruco_msgs::msg::MarkerArray>(
    "/detected_markers",
    10,
    markerCallback);

    // ID 1 마커가 저장될 때까지 대기 (최대 5초)
    rclcpp::Time start_time = node->now();
    while (rclcpp::ok() && !marker_saved && (node->now() - start_time).seconds() < 5.0) {
        rclcpp::spin_some(node);
        rclcpp::sleep_for(std::chrono::milliseconds(100));
    }

    if (!marker_saved) {
        RCLCPP_ERROR(node->get_logger(), "ID 1 아루코 마커를 찾지 못했습니다. 종료합니다.");
        rclcpp::shutdown();
        return 1;
    }


    // MoveIt! 그룹 설정
    moveit::planning_interface::MoveGroupInterface arm_group(node, "arm");
    moveit::planning_interface::MoveGroupInterface gripper_group(node, "gripper");

    rclcpp::sleep_for(std::chrono::seconds(2));

    // 그리퍼 열기
    gripper_group.setNamedTarget("open");
    moveit::planning_interface::MoveGroupInterface::Plan open_plan;
    if (gripper_group.plan(open_plan) == moveit::core::MoveItErrorCode::SUCCESS)
        gripper_group.execute(open_plan);

    rclcpp::sleep_for(std::chrono::seconds(1));
    
    // Arm의 현재 상태 가져오기
    auto current_state = arm_group.getCurrentState(3.0);
    
    //topic으로 받은 아루코 마커 정보
    // double x_ob=0.002683;
    // double y_ob=0.04857;
    // double z_ob=0.17891;

    Eigen::Vector3d tvec_optical(x_ob, y_ob, z_ob);
    Eigen::Vector3d tvec_in_eeframe = convertToBaseFrame(tvec_optical);

    //현재 조인트 각도에 따른 엔드이펙터의 위치
    //std::vector<double> joints = arm_group.getCurrentJointValues();
    Eigen::Matrix4d EE= forward_kinematics(0,DEG2RAD(0),DEG2RAD(-30),DEG2RAD(-30));
    //Eigen::Matrix4d EE= forward_kinematics(0,0,DEG2RAD(-30),DEG2RAD(-30));
    // Eigen::Matrix4d EE= forward_kinematics(joints[0],
    //     joints[1],
    //     joints[2],
    //     joints[3]);
    
    Eigen::Vector4d tvec_in_eeframe_4d(tvec_in_eeframe(0), tvec_in_eeframe(1), tvec_in_eeframe(2) , 1.0);
    Eigen::Vector4d TT=EE*tvec_in_eeframe_4d;
    
    //IK 진행
    Pose ee;
    //ee.px=tvec_in_base(0);
    //ee.py=tvec_in_base(1);
    //ee.pz=tvec_in_base(2);
    ee.px=TT(0);
    // if(x_ob>0)
    //      ee.py=-TT(1);
    // else
    //ee.py=TT(1);
    ee.py=0;
    ee.pz=TT(2)+0.03;
    //phi=theta2+theta3+theta4
    ee.phi=DEG2RAD(-60);

    RCLCPP_INFO(node->get_logger(), "Mark pose x=%.3f, y=%.3f, z=%.3f",x_ob, y_ob,z_ob);
    RCLCPP_INFO(node->get_logger(), "Obeject pose x=%.3f, y=%.3f, z=%.3f",ee.px, ee.py,ee.pz);

    IKResult joint=solveIK(ee,0.128,0.148,0.126,0.077,false);
    std::vector<double> joint_positions = {joint.theta1, joint.theta2, joint.theta3, joint.theta4};
    arm_group.setJointValueTarget(joint_positions);

    moveit::planning_interface::MoveGroupInterface::Plan approach_plan;
    if (arm_group.plan(approach_plan) == moveit::core::MoveItErrorCode::SUCCESS)
        arm_group.execute(approach_plan);


    rclcpp::sleep_for(std::chrono::seconds(1));
    
    // 그리퍼 닫기
    gripper_group.setJointValueTarget("gripper_left_joint", -0.002);
    moveit::planning_interface::MoveGroupInterface::Plan close_plan;
    if (gripper_group.plan(close_plan) == moveit::core::MoveItErrorCode::SUCCESS)
        gripper_group.execute(close_plan);

    rclcpp::sleep_for(std::chrono::seconds(1));

    // 물체를 놓을 위치로 이동
    std::vector<double> init_joint_positions = {
        -1.57,  // joint1
        0.8,   // joint2
        -0.3,  // joint3
        -0.4   // joint4
    };
    arm_group.setJointValueTarget(init_joint_positions);

    moveit::planning_interface::MoveGroupInterface::Plan init_plan;
    if (arm_group.plan(init_plan) == moveit::core::MoveItErrorCode::SUCCESS)
        arm_group.execute(init_plan);

    rclcpp::sleep_for(std::chrono::seconds(1));

    // 그리퍼 열기
    gripper_group.setNamedTarget("open");
    moveit::planning_interface::MoveGroupInterface::Plan final_open_plan;
    if (gripper_group.plan(final_open_plan) == moveit::core::MoveItErrorCode::SUCCESS)
        gripper_group.execute(final_open_plan);

    RCLCPP_INFO(node->get_logger(), "Pick and place finished!");

    rclcpp::sleep_for(std::chrono::seconds(1));

    // 초기 상태로 이동
     std::vector<double> place_joint_positions = {
        0,  // joint1
        DEG2RAD(0),   // joint2
        DEG2RAD(30),  // joint3
        DEG2RAD(30)   // joint4
    };
    arm_group.setJointValueTarget(place_joint_positions);

    moveit::planning_interface::MoveGroupInterface::Plan place_plan;
    if (arm_group.plan(place_plan) == moveit::core::MoveItErrorCode::SUCCESS)
        arm_group.execute(place_plan);

    rclcpp::shutdown();
    return 0;
}
