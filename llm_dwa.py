import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from api import get_waypoints
import time

class Robot:
    def __init__(self, x, y, theta, v=0.0, w=0.0):
        self.v = v  
        self.w = w
        self.x = x
        self.y = y
        self.theta = theta
        self.path_history = [(x, y)]  # List to record the robot's path history

class DWAConfig:
    def __init__(self):
        self.max_speed = 3.0        # Maximum speed
        self.min_speed = -0.5       # Minimum speed (including reverse)
        self.max_yaw_rate = np.deg2rad(45.0)
        self.max_accel = 1.0
        self.dt = 0.05              # Time step
        self.predict_time = 2.0     # Prediction time
        self.goal_cost_gain = 3.0
        self.heading_cost_gain = 1.0  # alpha
        self.obstacle_cost_gain = 1.0 # beta
        self.velocity_cost_gain = 0.5 # gamma


def add_map_border_obstacles(obstacles, grid_size, border_thickness=0.1):
    width, height = grid_size

    obstacles.append((0, 0, border_thickness, height))  # left
    obstacles.append((width - border_thickness, 0, border_thickness, height))  # right

    obstacles.append((0, 0, width, border_thickness))  # bottom
    obstacles.append((0, height - border_thickness, width, border_thickness))  # top

    return obstacles

def motion_model(robot, v, w, dt):
    robot.x += v * np.cos(robot.theta) * dt
    robot.y += v * np.sin(robot.theta) * dt
    robot.theta += w * dt
    robot.v = v
    robot.w = w
    robot.path_history.append((robot.x, robot.y))  # Add current position to path history

def calculate_dynamic_window(robot, config):
    predict_window_time = 0.5  # Range of speed change the robot can achieve in 0.5 seconds

    vs = [
        robot.v - config.max_accel * predict_window_time,
        robot.v + config.max_accel * predict_window_time
    ]
    ws = [
        robot.w - config.max_yaw_rate * predict_window_time,
        robot.w + config.max_yaw_rate * predict_window_time
    ]

    vs = np.clip(vs, config.min_speed, config.max_speed)
    ws = np.clip(ws, -config.max_yaw_rate, config.max_yaw_rate)

    return vs, ws


def is_robot_colliding(x, y, theta, obstacles, robot_size=0.5):
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

        if is_robot_colliding(traj_x, traj_y, traj_theta, obstacles):
            return float('inf'), np.array(trajectory)

        goal_cost = np.hypot(goal[0] - traj_x, goal[1] - traj_y)

        desired_angle = np.arctan2(goal[1] - traj_y, goal[0] - traj_x)
        heading_error = abs(np.arctan2(np.sin(traj_theta - desired_angle), np.cos(traj_theta - desired_angle)))

        min_dist_to_obstacle = min([np.hypot(traj_x - ox, traj_y - oy) for ox, oy, _, _ in obstacles])
        clearance_cost = 1.0 / (min_dist_to_obstacle + 1e-6)

        velocity_cost = config.max_speed - v

        cost += config.goal_cost_gain * goal_cost \
                + config.heading_cost_gain * heading_error \
                + config.obstacle_cost_gain * clearance_cost \
                + config.velocity_cost_gain * velocity_cost

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

def calculate_path_length(path_history):
    length = 0.0
    for i in range(1, len(path_history)):
        x1, y1 = path_history[i - 1]
        x2, y2 = path_history[i]
        length += np.hypot(x2 - x1, y2 - y1)
    return length

def plot_robot(robot, start, goal, current_target, obstacles, grid_size, reached_target, trajectories=None, chosen_trajectory=None, robot_size=0.5):
    plt.cla()
    ax = plt.gca()
    half = robot_size / 2.0
    for obs in obstacles:
        ox, oy, w, h = obs
        ax.add_patch(plt.Rectangle((ox, oy), w, h, color='black', alpha=1.0))
    
    plt.scatter(start[0], start[1], color='blue', s=100, marker='o', label="Start")
    plt.scatter(goal[0], goal[1], color='red', s=100, marker='s', label="Goal")
    if reached_target:
        for pt in reached_target:
            plt.scatter(pt[0], pt[1], color='purple', s=100, marker='*')
    plt.scatter(current_target[0], current_target[1], color='green', s=100, marker='*', label="Current Target")
    
    if len(robot.path_history) > 1:
        path_x, path_y = zip(*robot.path_history)
        plt.plot(path_x, path_y, linestyle='--', color='orange', linewidth=2, label='Robot Path')
    
    corners = []
    for dx, dy in [(-half, -half), (-half, half), (half, half), (half, -half)]:
        x = robot.x + (dx * np.cos(robot.theta) - dy * np.sin(robot.theta))
        y = robot.y + (dx * np.sin(robot.theta) + dy * np.cos(robot.theta))
        corners.append((x, y))

    robot_poly = Polygon(corners, closed=True, color='blue', alpha=1.0)
    ax.add_patch(robot_poly)

    # Plot possible trajectories
    if trajectories:
        for traj in trajectories:
            plt.plot(traj[:, 0], traj[:, 1], color='yellow', alpha=0.5, linewidth=1)
    
    # Plot chosen trajectory
    if chosen_trajectory is not None:
        plt.plot(chosen_trajectory[:, 0], chosen_trajectory[:, 1], color='blue', alpha=1.0, linewidth=1.5)
    
    plt.xlim(0, grid_size[0])
    plt.ylim(0, grid_size[1])
    plt.grid(True)
    plt.legend()
    plt.draw()
    plt.pause(0.01)

def main():
    start_time = time.time()
    grid_size = (20, 20)
    start = (3, 7)
    goal = (1, 19)
    obstacles = [
        (6, 5, 1, 6),
        (6, 5, 7, 1),
        (12, 5, 1, 5),
        (18, 3, 1, 5),
        (9, 10, 1, 4),
        (0, 13, 13, 1),
        (2, 18, 4, 1)
    ]
    
    obstacles = add_map_border_obstacles(obstacles, grid_size)
    result = get_waypoints(grid_size, start, goal, obstacles)
    if isinstance(result, dict):
        waypoints = result.get("waypoints")
    else:
        waypoints = result
    
    if not waypoints:
        print("Cannot generate waypoints.")
        return
    
    robot = Robot(*waypoints[0], np.pi / 4)
    config = DWAConfig()

    target_idx = 1
    current_target = waypoints[target_idx]
    reached_target = []  
    
    plt.figure(figsize=(8, 8))
    
    for _ in range(1000):
        v, w, trajectories, best_trajectory = dwa_control(robot, config, current_target, obstacles)
        motion_model(robot, v, w, config.dt)
        plot_robot(robot, start, goal, current_target, obstacles, grid_size, reached_target, trajectories, best_trajectory)
        
        distance_to_target = np.hypot(robot.x - current_target[0], robot.y - current_target[1])
        if distance_to_target < 0.5:
            print(f"Reached waypoint {target_idx}: {current_target}")
            reached_target.append(current_target)
            target_idx += 1
            if target_idx >= len(waypoints):
                print("Reached the final goal!")
                break
            current_target = waypoints[target_idx]

    total_distance = calculate_path_length(robot.path_history)
    print(f"Total travel distance: {total_distance:.2f} m")
    end_time = time.time()
    print(f'total_time: {end_time - start_time:.2f} seconds')
    plt.show()

if __name__ == "__main__":
    main()