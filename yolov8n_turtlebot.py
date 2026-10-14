#!/usr/bin/env python3
import heapq
from math import atan2, cos, floor, hypot, sin, sqrt
import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    Twist,
)
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


def octile_distance(x, y, gx, gy):
    dx = abs(gx - x)
    dy = abs(gy - y)
    return max(dx, dy) + (sqrt(2.0) - 1.0) * min(dx, dy)


def a_star(grid, start, goal):
    rows, cols = grid.shape
    sx, sy = start
    gx, gy = goal

    if not (0 <= sx < cols and 0 <= sy < rows):
        return None, np.inf
    if not (0 <= gx < cols and 0 <= gy < rows):
        return None, np.inf

    # 0: drivable
    if grid[sy, sx] != 0 or grid[gy, gx] != 0:
        return None, np.inf

    directions = [
        (0, 1, 1.0), (0, -1, 1.0),
        (1, 0, 1.0), (-1, 0, 1.0),
        (1, 1, sqrt(2.0)), (1, -1, sqrt(2.0)),
        (-1, 1, sqrt(2.0)), (-1, -1, sqrt(2.0)),
    ]

    distance = np.full((rows, cols), np.inf)
    distance[sy, sx] = 0.0

    parent = np.full((rows, cols, 2), -1, dtype=int)
    visited = np.zeros((rows, cols), dtype=bool)

    queue = [(octile_distance(sx, sy, gx, gy), sx, sy)]

    while queue:
        _, x, y = heapq.heappop(queue)

        if visited[y, x]:
            continue

        visited[y, x] = True

        if (x, y) == (gx, gy):
            break

        for dx, dy, step_cost in directions:
            nx, ny = x + dx, y + dy

            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if grid[ny, nx] != 0 or visited[ny, nx]:
                continue

            # save corner pass
            if dx != 0 and dy != 0:
                if grid[y, nx] != 0 or grid[ny, x] != 0:
                    continue

            new_cost = distance[y, x] + step_cost

            if new_cost < distance[ny, nx]:
                distance[ny, nx] = new_cost
                parent[ny, nx] = [x, y]

                h = octile_distance(nx, ny, gx, gy)
                heapq.heappush(queue, (new_cost + h, nx, ny))

    if not visited[gy, gx]:
        return None, np.inf

    path = []
    current = (gx, gy)

    while current != (sx, sy):
        path.append(current)
        px, py = parent[current[1], current[0]]

        if px == -1:
            return None, np.inf

        current = (int(px), int(py))

    path.append((sx, sy))
    path.reverse()

    return path, distance[gy, gx]


class Navigation(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Map
        self.map_data = None
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.map_origin_yaw = 0.0

        # Robot pose
        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.is_localized = False

        # Goal and path
        self.goal_pose = None
        self.plan_requested = False
        self.path = []
        self.current_waypoint_index = 0
        self.has_path = False

        # Controller parameters
        self.lookahead_distance = 0.5
        self.linear_velocity = 0.1
        self.goal_tolerance = 0.15
        self.max_angular_velocity = 1.0

        # YOLO & Object Detection
        self.bridge = CvBridge()
        self.yolo_model = YOLO('yolov8n.pt')
        self.is_object_detected = False

        # Specify target objects that trigger emergency stop
        self.target_objects = ['keyboard']

        # Map server commonly publishes TRANSIENT_LOCAL maps
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10
        )

        # Subscription for Camera Stream
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/reference_path', 10)
        self.detection_pub = self.create_publisher(String, '/yolo/detected_objects', 10)

        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(
            'Navigation node started. Waiting for map, pose and goal...'
        )

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        results = self.yolo_model(cv_image, conf=0.5, verbose=False)
        detected_names = []
        target_found = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = self.yolo_model.names[cls]
                detected_names.append(name)

                # Check if detected object matches target objects
                if name in self.target_objects:
                    target_found = True
                    color = (0, 0, 255)  # Red box for target objects
                else:
                    color = (127, 0, 127)  # Purple box for other objects

                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    cv_image,
                    f"{name} {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        self.is_object_detected = target_found

        if detected_names:
            msg_out = String()
            msg_out.data = ", ".join(detected_names)
            self.detection_pub.publish(msg_out)

        cv2.imshow("TurtleBot YOLO Detection", cv_image)
        cv2.waitKey(1)

    def map_callback(self, msg):
        self.map_data = np.asarray(
            msg.data, dtype=np.int16
        ).reshape((msg.info.height, msg.info.width))

        self.map_resolution = msg.info.resolution

        origin = msg.info.origin
        self.map_origin_x = origin.position.x
        self.map_origin_y = origin.position.y

        q = origin.orientation
        self.map_origin_yaw = atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        self.get_logger().info(
            f'Map received: {msg.info.width} x {msg.info.height}, '
            f'resolution={self.map_resolution:.3f} m'
        )
        self.try_plan()

    def pose_callback(self, msg):
        pose = msg.pose.pose
        self.current_x = pose.position.x
        self.current_y = pose.position.y

        q = pose.orientation
        self.current_yaw = atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        if not self.is_localized:
            self.get_logger().info('AMCL pose received.')

        self.is_localized = True
        self.try_plan()

    def goal_callback(self, msg):
        self.goal_pose = (
            msg.pose.position.x,
            msg.pose.position.y,
        )

        self.has_path = False
        self.path = []
        self.current_waypoint_index = 0
        self.plan_requested = True
        self.stop_robot()

        self.get_logger().info(
            f'New goal: x={self.goal_pose[0]:.3f}, '
            f'y={self.goal_pose[1]:.3f}'
        )
        self.try_plan()

    def world_to_grid(self, position):
        x, y = position
        dx = x - self.map_origin_x
        dy = y - self.map_origin_y

        local_x = (
            cos(self.map_origin_yaw) * dx
            + sin(self.map_origin_yaw) * dy
        )
        local_y = (
            -sin(self.map_origin_yaw) * dx
            + cos(self.map_origin_yaw) * dy
        )

        return (
            floor(local_x / self.map_resolution),
            floor(local_y / self.map_resolution),
        )

    def grid_to_world(self, point):
        gx, gy = point

        local_x = (gx + 0.5) * self.map_resolution
        local_y = (gy + 0.5) * self.map_resolution

        x = (
            self.map_origin_x
            + cos(self.map_origin_yaw) * local_x
            - sin(self.map_origin_yaw) * local_y
        )
        y = (
            self.map_origin_y
            + sin(self.map_origin_yaw) * local_x
            + cos(self.map_origin_yaw) * local_y
        )

        return x, y

    def publish_path(self):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for x, y in self.path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)

    def try_plan(self):
        if not self.plan_requested:
            return
        if self.map_data is None:
            return
        if not self.is_localized or self.goal_pose is None:
            return

        start = self.world_to_grid((self.current_x, self.current_y))
        goal = self.world_to_grid(self.goal_pose)
        rows, cols = self.map_data.shape

        self.get_logger().info(
            f'Planning A*: start={start}, goal={goal}, '
            f'map_size={cols}x{rows}'
        )

        for name, (gx, gy) in [('start', start), ('goal', goal)]:
            if not (0 <= gx < cols and 0 <= gy < rows):
                self.plan_requested = False
                self.has_path = False
                self.stop_robot()
                self.get_logger().error(f'{name} outside map: {(gx, gy)}')
                return

        grid_path, path_cost = a_star(self.map_data, start, goal)
        self.plan_requested = False

        if grid_path is None:
            self.has_path = False
            self.path = []
            self.publish_path()
            self.stop_robot()
            self.get_logger().error(
                'A*: No valid path found. Check occupancy of start/goal.'
            )
            return

        self.path = [
            list(self.grid_to_world(point)) for point in grid_path
        ]
        self.current_waypoint_index = 0
        self.has_path = True

        # Publish the generated path for RViz
        self.publish_path()

        final_x, final_y = self.path[-1]
        goal_error = hypot(
            final_x - self.goal_pose[0],
            final_y - self.goal_pose[1],
        )

        self.get_logger().info(
            f'A*: {len(self.path)} waypoints, '
            f'cost={path_cost:.2f} cells, '
            f'approx. length={path_cost * self.map_resolution:.2f} m'
        )
        self.get_logger().info(
            f'Goal=({self.goal_pose[0]:.3f}, {self.goal_pose[1]:.3f}), '
            f'last waypoint=({final_x:.3f}, {final_y:.3f}), '
            f'cell-center error={goal_error:.3f} m'
        )

    def find_lookahead_point(self):
        # Find a path point at least lookahead_distance away.
        for i in range(self.current_waypoint_index, len(self.path)):
            x, y = self.path[i]
            if hypot(x - self.current_x, y - self.current_y) >= self.lookahead_distance:
                self.current_waypoint_index = i
                return x, y

        return self.path[-1]

    def control_loop(self):
        if not self.is_localized or not self.has_path or not self.path:
            return

        # Emergency stop if target object is detected by YOLO
        if self.is_object_detected:
            self.stop_robot()
            return

        final_x, final_y = self.path[-1]
        if hypot(
            final_x - self.current_x,
            final_y - self.current_y,
        ) < self.goal_tolerance:
            self.has_path = False
            self.stop_robot()
            self.get_logger().info('Goal reached; stopping robot.')
            return

        target_x, target_y = self.find_lookahead_point()
        dx = target_x - self.current_x
        dy = target_y - self.current_y

        target_angle = atan2(dy, dx)
        alpha = atan2(
            sin(target_angle - self.current_yaw),
            cos(target_angle - self.current_yaw),
        )

        curvature = (
            2.0 * sin(alpha) / max(self.lookahead_distance, 1e-6)
        )

        cmd = Twist()
        cmd.linear.x = self.linear_velocity
        cmd.angular.z = max(
            -self.max_angular_velocity,
            min(
                self.max_angular_velocity,
                self.linear_velocity * curvature,
            ),
        )
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = Navigation()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
