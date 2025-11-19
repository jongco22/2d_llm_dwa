import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math

# DWA와 Dijkstra를 통합한 하이브리드 경로 계획 시뮬레이터

# ==============================================================================
# Dijkstra 알고리즘 코드 부분 (Global Path Planner)
# ==============================================================================
class DijkstraPlanner:
    """
    Grid based Dijkstra planning
    """

    def __init__(self, ox, oy, resolution, robot_radius):
        """
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        robot_radius: robot radius[m]
        """
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.x_width = None
        self.y_width = None
        self.obstacle_map = None

        self.resolution = resolution
        self.robot_radius = robot_radius
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

    def planning(self, sx, sy, gx, gy):
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_index(start_node)] = start_node

        while True:
            if not open_set:
                print("Cannot find path")
                return [], []
                
            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost, c_id)
                n_id = self.calc_index(node)

                if n_id in closed_set or not self.verify_node(node):
                    continue

                if n_id not in open_set or open_set[n_id].cost > node.cost:
                    open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        rx, ry = [self.calc_position(goal_node.x, self.min_x)], [
            self.calc_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_position(n.x, self.min_x))
            ry.append(self.calc_position(n.y, self.min_y))
            parent_index = n.parent_index
        return rx, ry

    def calc_position(self, index, minp):
        return index * self.resolution + minp

    def calc_xy_index(self, position, minp):
        return round((position - minp) / self.resolution)

    def calc_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_position(node.x, self.min_x)
        py = self.calc_position(node.y, self.min_y)

        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
            return False
        if self.obstacle_map[node.x][node.y]:
            return False
        return True

    def calc_obstacle_map(self, ox, oy):
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        self.obstacle_map = [[False for _ in range(self.y_width)] for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    if math.hypot(iox - x, ioy - y) <= self.robot_radius:
                        self.obstacle_map[ix][iy] = True
                        break
    
    @staticmethod
    def get_motion_model():
        return [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
                [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]

# ==============================================================================
# DWA 알고리즘 코드 부분 (Local Path Planner)
# ==============================================================================
class Robot:
    def __init__(self, x, y, theta, v=0.0, w=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.w = w
        self.path_history = [(x, y)]

class DWAConfig:
    def __init__(self):
        self.max_speed = 1.0
        self.min_speed = -0.5
        self.max_yaw_rate = np.deg2rad(45.0)
        self.max_accel = 0.4
        self.dt = 0.05
        self.predict_time = 3.0
        self.goal_cost_gain = 3.0
        self.heading_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.velocity_cost_gain = 0.5
        self.robot_size = 0.5 # For collision checking

def motion_model(robot, v, w, dt):
    robot.x += v * np.cos(robot.theta) * dt
    robot.y += v * np.sin(robot.theta) * dt
    robot.theta += w * dt
    robot.v = v
    robot.w = w
    robot.path_history.append((robot.x, robot.y))

def calculate_dynamic_window(robot, config):
    predict_window_time = 0.5
    vs = [robot.v - config.max_accel * predict_window_time,
          robot.v + config.max_accel * predict_window_time]
    ws = [robot.w - config.max_yaw_rate * predict_window_time,
          robot.w + config.max_yaw_rate * predict_window_time]
    
    vs = np.clip(vs, config.min_speed, config.max_speed)
    ws = np.clip(ws, -config.max_yaw_rate, config.max_yaw_rate)
    return vs, ws

def is_robot_colliding(x, y, theta, obstacles, robot_size):
    half = robot_size / 2.0
    corners = []
    for dx, dy in [(-half, -half), (-half, half), (half, half), (half, -half)]:
        cx = x + (dx * np.cos(theta) - dy * np.sin(theta))
        cy = y + (dx * np.sin(theta) + dy * np.cos(theta))
        corners.append((cx, cy))

    for cx, cy in corners:
        for ox, oy, w, h in obstacles:
            if ox <= cx <= ox + w and oy <= cy <= oy + h:
                return True
    return False

def evaluate_trajectory(robot, v, w, config, goal, obstacles):
    traj_x, traj_y, traj_theta = robot.x, robot.y, robot.theta
    trajectory = []
    cost = 0.0
    steps = int(config.predict_time / config.dt)

    for _ in range(steps):
        traj_x += v * np.cos(traj_theta) * config.dt
        traj_y += v * np.sin(traj_theta) * config.dt
        traj_theta += w * config.dt
        trajectory.append([traj_x, traj_y])

        if is_robot_colliding(traj_x, traj_y, traj_theta, obstacles, config.robot_size):
            return float('inf'), np.array(trajectory)

    # 마지막 지점만으로 평가
    final_x, final_y = trajectory[-1]
    goal_cost = np.hypot(goal[0] - final_x, goal[1] - final_y)
    
    desired_angle = np.arctan2(goal[1] - final_y, goal[0] - final_x)
    heading_error = abs(np.arctan2(np.sin(traj_theta - desired_angle), np.cos(traj_theta - desired_angle)))

    min_dist_to_obstacle = float('inf')
    for ox, oy, w, h in obstacles:
        # 직사각형 장애물의 중심점과의 거리로 단순화
        center_x, center_y = ox + w/2, oy + h/2
        dist = np.hypot(final_x - center_x, final_y - center_y)
        min_dist_to_obstacle = min(min_dist_to_obstacle, dist)
    clearance_cost = 1.0 / (min_dist_to_obstacle + 1e-6)

    velocity_cost = config.max_speed - v

    cost = (config.goal_cost_gain * goal_cost +
            config.heading_cost_gain * heading_error +
            config.obstacle_cost_gain * clearance_cost +
            config.velocity_cost_gain * velocity_cost)

    return cost, np.array(trajectory)


def dwa_control(robot, config, goal, obstacles):
    best_v, best_w = 0.0, 0.0
    min_cost = float('inf')
    best_trajectory = None
    trajectories = []

    v_range, w_range = calculate_dynamic_window(robot, config)

    for v in np.arange(v_range[0], v_range[1] + 0.01, 0.1):
        for w in np.arange(w_range[0], w_range[1] + 0.01, np.deg2rad(2.5)):
            cost, trajectory = evaluate_trajectory(robot, v, w, config, goal, obstacles)
            trajectories.append(trajectory)
            if cost < min_cost:
                min_cost = cost
                best_v, best_w = v, w
                best_trajectory = trajectory

    return best_v, best_w, trajectories, best_trajectory

def plot_simulation(robot, start, goal, obstacles, grid_size, global_path, trajectories=None, chosen_trajectory=None):
    plt.cla()
    ax = plt.gca()

    # 장애물 그리기
    for obs in obstacles:
        ox, oy, w, h = obs
        ax.add_patch(plt.Rectangle((ox, oy), w, h, color='black'))

    # 시작점, 목표점 그리기
    plt.scatter(start[0], start[1], color='blue', s=100, marker='o', label="Start")
    plt.scatter(goal[0], goal[1], color='red', s=100, marker='s', label="Final Goal")
    
    # Dijkstra 전역 경로 그리기
    if global_path:
        gx, gy = zip(*global_path)
        plt.plot(gx, gy, linestyle='-', color='green', linewidth=3, label='Global Path (Dijkstra)')

    # 로봇의 실제 이동 경로 그리기
    if len(robot.path_history) > 1:
        path_x, path_y = zip(*robot.path_history)
        plt.plot(path_x, path_y, linestyle='--', color='orange', linewidth=2, label='Robot Path (DWA)')
    
    # DWA 후보 경로들 그리기
    if trajectories:
        for traj in trajectories:
            plt.plot(traj[:, 0], traj[:, 1], color='yellow', alpha=0.3, linewidth=1)

    # DWA가 선택한 최적 경로 그리기
    if chosen_trajectory is not None:
        plt.plot(chosen_trajectory[:, 0], chosen_trajectory[:, 1], color='blue', alpha=0.8, linewidth=1.5)

    # 로봇 그리기
    half = DWAConfig().robot_size / 2.0
    corners = []
    for dx, dy in [(-half, -half), (-half, half), (half, half), (half, -half)]:
        x = robot.x + (dx * np.cos(robot.theta) - dy * np.sin(robot.theta))
        y = robot.y + (dx * np.sin(robot.theta) + dy * np.cos(robot.theta))
        corners.append((x, y))
    ax.add_patch(Polygon(corners, closed=True, color='blue', alpha=0.8))

    plt.xlim(0, grid_size[0])
    plt.ylim(0, grid_size[1])
    plt.grid(True)
    plt.legend()
    plt.pause(0.01)

# ==============================================================================
# 메인 실행 함수
# ==============================================================================
def main():
    print("Dijkstra + DWA Hybrid Path Planning start!!")
    
    # --- 시뮬레이션 설정 ---
    grid_size = (20, 20)
    start = (3, 7)
    final_goal = (1, 19)
    obstacles_rect = [
        (6, 5, 1, 6), (6, 5, 7, 1), (12, 5, 1, 5), (18, 3, 1, 5),
        (9, 10, 1, 4), (0, 13, 13, 1), (2, 18, 4, 1)
    ]
    
    # --- 1. Dijkstra로 전역 경로 생성 ---
    # DWA 장애물(사각형)을 Dijkstra 장애물(점)로 변환
    ox, oy = [], []
    for obs in obstacles_rect:
        x, y, w, h = obs
        # 사각형의 경계와 내부를 점으로 샘플링
        for i in np.arange(x, x + w, 0.5):
            for j in np.arange(y, y + h, 0.5):
                ox.append(i)
                oy.append(j)
    # 맵 경계도 장애물로 추가
    for i in np.arange(0, grid_size[0], 0.5):
        ox.extend([i, i, 0, grid_size[0]])
        oy.extend([0, grid_size[1], i, i])

    # Dijkstra 플래너 초기화 및 경로 계산
    dijkstra_resolution = 0.5
    dijkstra_robot_radius = 0.8
    dijkstra = DijkstraPlanner(ox, oy, dijkstra_resolution, dijkstra_robot_radius)
    rx, ry = dijkstra.planning(start[0], start[1], final_goal[0], final_goal[1])
    
    if not rx:
        print("Dijkstra failed to find a path.")
        return

    # 경로를 시작점에서 목표점 순으로 뒤집기
    global_path = list(zip(rx, ry))[::-1]

    # --- 2. DWA로 지역 경로 주행 ---
    robot = Robot(start[0], start[1], np.pi / 4)
    config = DWAConfig()
    
    # DWA가 따라갈 경유지 인덱스
    target_waypoint_index = 0
    
    plt.figure(figsize=(10, 10))

    for i in range(1000):
        # 현재 목표 경유지 설정
        current_target = global_path[target_waypoint_index]

        # DWA 제어 실행
        v, w, trajectories, best_trajectory = dwa_control(robot, config, current_target, obstacles_rect)
        
        # 로봇 상태 업데이트
        motion_model(robot, v, w, config.dt)
        
        # 시각화
        plot_simulation(robot, start, final_goal, obstacles_rect, grid_size, global_path, trajectories, best_trajectory)

        # 현재 경유지에 도달했는지 확인
        dist_to_waypoint = np.hypot(robot.x - current_target[0], robot.y - current_target[1])
        if dist_to_waypoint < 1.0: # 경유지 도착 반경
            # 마지막 경유지가 아니면 다음 경유지로 업데이트
            if target_waypoint_index < len(global_path) - 1:
                target_waypoint_index += 1
            else:
                 # 최종 목표점에 가까워지면 루프 탈출 조건 확인
                if np.hypot(robot.x - final_goal[0], robot.y - final_goal[1]) < 0.5:
                    print("Goal reached!")
                    break
        
        # 최종 목표 도달 확인 (백업)
        if np.hypot(robot.x - final_goal[0], robot.y - final_goal[1]) < 0.5:
            print("Goal reached!")
            break
            
    print("Simulation finished.")
    plt.show()

if __name__ == "__main__":
    main()
