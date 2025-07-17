<h1 align="center">디지털 트윈 기반 서비스 로봇 운영 시스템📟 </h1>

<h2 align="center">라인 기반 경로 추종과 물체 이송이 가능한 로봇 시스템🚜 </h2>



## 개요

창고 내 물품 운반 및 분류를 목적으로 함
소형 자율주행 로봇이 창고 내 지정된 경로를 따라 이동하여 yolo를 이용해 특정 물품이나 표지판을 인식 가능, 소규모 창고에서 저비용 자동화 구현 (ex.아마존 물류창고 Kiva robot)






## 제작 기간 & 참여 인원


-2025/05/09~2025/05/22  5명






## 사용한 기술 (기술 스택)  


<img src="https://img.shields.io/badge/python-blue?style=for-the-badge&logo=python&logoColor=white">   <img src="https://img.shields.io/badge/ROS2-black?style=for-the-badge&logo=ros&logoColor=#22314E">   <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">   <img src="https://img.shields.io/badge/YOLO-111F68?style=for-the-badge&logo=yolo&logoColor=white">  <img src="https://img.shields.io/badge/Jetson-green?style=for-the-badge&logo=Jetson&logoColor=white">  <img src="https://img.shields.io/badge/Aruco-blue?style=for-the-badge&logo=Aruco&logoColor=white">  <img src="https://img.shields.io/badge/Gazebo-red?style=for-the-badge&logo=Gazebo&logoColor=white">



## 내가 기여한 부분


- **YOLO 학습 및 Jetson Orin 연결**




## 🌟핵심 기능 (코드로 보여주거나 코드 링크)







   
## 🎯트러블슈팅 경험  


1. Moveit2를 사용해 getCurrentPose()로 End-Effector의 현재 위치(Pose) 확인이 사용 불가, setPositionTarget()로 목표 위치 지정하려했지만 Moveit2 사용 실패
 


2. Lane detect를 할 때 불필요한 배경(벽, 천장 등)에서 오는 오탐지를 줄이고, 연산 속도를 높이기 위해 ROI(관심 영역) 설정을 통해 이미지의 하단 40%만 사용하여 중앙 노란색 선을 계산하니 깊은 코너에서 중앙 실선이 범위에서 벗어나는 문제 발생



## 🔨해결방법


1-1. 직접 해석적(Analytical) 역기구학 (IK) 구현
MoveIt 대신 직접 IK 수식을 유도하여 C++ 코드로 구현하고 End-Effector의 위치(x, y, z)와 pitch 각도만 고려하여 가능한 해를 계산
계산된 관절 각도를 MoveIt의 setJointValueTarget()으로 전달해 실행


2-1. 좌 우측 30%를 ROI로 지정 후 해당 ROI에 실선이 인식될 시 해당 방향으로 주행하도록 하여 문제를 해결
이 방법으로 카메라에 중앙에서 갑작스레 벗어나는 직각 실선에서도 원활한 주행이 가능


## 회고 / 느낀 점

-ROS2를 사용한 첫 프로젝트라서 interface를 제대로 활용하지 못했다는 생각이 들었다. 이에 대한 공부가 부족하다고 느껴 이론 공부를 추가적으로 진행했다.

-moveit2를 사용하지 못 한 원인을 파악하지 못 하고 프로젝트가 종료되어 아쉬웠다.
