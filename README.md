# ROS 2 A* Navigation with YOLOv8

ROS 2 기반의 **A* 알고리즘 경로 계획**, **Pure Pursuit 경로 추종**, 그리고 **YOLOv8 객체 인식 기반 비상 정지**를 수행하는 자율주행 노드입니다.

---

## 🌟 주요 기능

- **A* Path Planning**: Octile distance 휴리스틱 및 모퉁이 충돌 방지 로직이 적용된 8방향 경로 생성
- **Pure Pursuit Control**: 생성된 경로를 따라 로봇을 안정적으로 추종 제어
- **YOLOv8 Object Detection**: 실시간 카메라 영상 분석 및 지정된 장애물(예: `keyboard`) 감지 시 비상 정지(Emergency Stop)

---

## 🛠 필수 라이브러리 (Dependencies)

- ROS 2 (Humble 이상 권장)
- Python 3.8+
- OpenCV & cv_bridge
- Ultralytics (YOLOv8)

```bash
pip install ultralytics
```

---

## 📡 토픽 (Topics)

### Subscribed Topics
- `/map` (`nav_msgs/msg/OccupancyGrid`) : 점유 격자 지도
- `/amcl_pose` (`geometry_msgs/msg/PoseWithCovarianceStamped`) : 로봇 현재 위치
- `/goal_pose` (`geometry_msgs/msg/PoseStamped`) : RViz2 등을 통한 목표 위치
- `/image_raw` (`sensor_msgs/msg/Image`) : 카메라 영상 스트림

### Published Topics
- `/cmd_vel` (`geometry_msgs/msg/Twist`) : 로봇 이동 제어 명령
- `/reference_path` (`nav_msgs/msg/Path`) : 생성된 A* 경로 (RViz 시각화용)
- `/yolo/detected_objects` (`std_msgs/msg/String`) : 검출된 객체 이름 목록

---

## 🚀 실행 방법

```bash
python3 navigation_node.py
```
