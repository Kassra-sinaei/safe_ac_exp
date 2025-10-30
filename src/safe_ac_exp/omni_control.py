#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from types import SimpleNamespace
import scipy.sparse as sparse
from scipy.spatial.transform import Rotation
from safe_ac_exp.controller import Controller
import osqp
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from vicon_receiver.msg import Position

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'text.usetex': True,  
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

class MobileRobotController(Controller, Node):
    """ 
    Mobile robot controller with adaptive CBF-QP.
    It contains the main control loop and the QP solver.
    """
    def __init__(self, settings, **controller_params):
        Controller.__init__(self, **controller_params)
        Node.__init__(self, "mobile_robot_controller")
        
        # Controller settings
        self.settings = settings
        self.kappa = settings['kappa']

        # Obstacle definitions
        self.obstacles = [
            {'center': np.array([[2.0], [1.5]]), 'radius': 0.45},
            {'center': np.array([[3.0], [2.5]]), 'radius': 0.45}
        ]

        # SI to Unicycle conversion parameters
        self.yaw = None

        # Vicon Coordinate frame correction
        self.theta = -np.pi/2
        self.mocap_R = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        self.mocap_P = None

        # Trajectory index
        self.index = 0
        self.experiment_complete = False

        # RBF basis function setup
        num_centers_per_dim = 5
        x_centers = np.linspace(settings['workspace'][0], settings['workspace'][1], num_centers_per_dim)
        y_centers = np.linspace(settings['workspace'][2], settings['workspace'][3], num_centers_per_dim)
        z_centers = np.linspace(settings['workspace'][4], settings['workspace'][5], num_centers_per_dim)
        xv, yv, zv = np.meshgrid(x_centers, y_centers, z_centers)
        self.rbf_centers = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()])
        self.rbf_width = (settings['workspace'][1] - settings['workspace'][0]) / (num_centers_per_dim - 1)

        print(f"RBF Centers shape: {self.rbf_centers.shape}")
        print(f"RBF Width: {self.rbf_width}")

        # ROS Setup
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.mocap_topics = {
            '/vicon/go_1/go_1': 'robot',
            '/vicon/box_1/box_1': 'obs1',
            '/vicon/box_2/box_2': 'obs2'
        }
        self.vicon_subs = []
        for topic in self.mocap_topics.keys():
            sub = self.create_subscription(
                Position,
                topic,
                self.mocap_callback,
                10
            )
            self.vicon_subs.append(sub)
        self.timer_ = self.create_timer(settings['dt'], self.control_loop)
        self.msg_ = Twist()


    def z(self, x):
        """
        Provides the Radial Basis Function (RBF) vector.
        x could be only the a function of time or the full state.
        in case it is full state use self.x
        """
        diffs = self.x - self.rbf_centers
        rbf_vals = np.exp(-np.sum(diffs**2, axis=0) / (2 * self.rbf_width**2))
        rbf_vec = np.hstack([1, rbf_vals])

        Z = np.kron(np.eye(self.n_x), rbf_vec)  
        return Z

    def cbf(self):
        h_1 = np.linalg.norm(self.x[0:2] - self.settings['obs1']['center'])**2 - self.settings['obs1']['radius'] ** 2
        h_2 = np.linalg.norm(self.x[0:2] - self.settings['obs2']['center'])**2 - self.settings['obs2']['radius'] ** 2

        return -1/self.kappa * np.log(np.exp(-self.kappa * h_1) + np.exp(-self.kappa * h_2))

    def cbf_grad(self):
        h_1 = np.linalg.norm(self.x[0:2] - self.settings['obs1']['center'])**2 - self.settings['obs1']['radius'] ** 2
        h_2 = np.linalg.norm(self.x[0:2] - self.settings['obs2']['center'])**2 - self.settings['obs2']['radius'] ** 2

        g_1 = 2 * (self.x[0:2] - self.settings['obs1']['center']).T 
        g_2 = 2 * (self.x[0:2] - self.settings['obs2']['center']).T 
        lambda_1 = np.exp(-self.kappa * h_1)
        lambda_2 = np.exp(-self.kappa * h_2)
        return np.hstack((lambda_1 * g_1 + lambda_2 * g_2, np.array([[0.0]])))

    def v_a_partial_e(self, error):
        return error.T

    def v_a_partial_w(self, error):
        return np.zeros((1, self.n_p))

    def alpha_3(self, error):
        return self.K_e * np.linalg.norm(error)**2

    def u(self, x_d, xd_d, time):
        """
        Adaptive control law with body-frame transformation for omni-directional robot.
        Overrides parent Controller.u() to account for heading angle coupling.
        """
        x_d = x_d.reshape(self.n_x, 1)
        xd_d = xd_d.reshape(self.n_x, 1)
        
        # Compute world-frame error
        error_world = self.x - x_d
        
        # Transform position error to body frame using current heading
        cos_theta = np.cos(self.x[2, 0])
        sin_theta = np.sin(self.x[2, 0])
        R_world_to_body = np.array([[cos_theta, sin_theta, 0],
                                     [-sin_theta,  cos_theta, 0],
                                     [0,          0,         1]])
        
        # Error in body frame 
        error_body = R_world_to_body @ error_world
        # Desired velocity in body frame
        xd_d_body = R_world_to_body @ xd_d

        # Update integral for bounds
        self.int_err_sq += (error_world.T @ error_world).item() * self.dt
        
        # Update parameter estimate (still uses world-frame error for stability proof)
        self.update(error_world, time)
        
        # Solve CBF-QP for safe parameter modification
        delta_w = self.cbf_qp(error_world, xd_d_body, time)
        
        if delta_w is None:
            print("\nERROR\nQP failed, using zero delta_w\n")
            delta_w = np.zeros((self.n_p, 1))
            u = self.saturate(self.z(time) @ self.w + xd_d_body - self.K @ error_body)
        else:
            # Apply control in body frame
            u = self.z(time) @ (self.w + delta_w) + xd_d_body - self.K @ error_body

        # Store trajectories
        self.w_traj.append(self.w.flatten())
        self.u_traj.append(u.flatten())
        self.joint_traj.append(self.x.flatten())
        self.vel_traj.append(self.x_d.flatten())
        self.e_traj.append(error_world.flatten())
        
        return u.flatten().tolist()
    
    def control_loop(self):
        if self.yaw == None:
            return
        if self.index > len(self.settings['time']) - 1:
            self.get_logger().info('Reached the end of the trajectory. Stopping the robot.')
            self.msg_.linear.x = 0.0
            self.msg_.angular.z = 0.0
            self.msg_.linear.y = 0.0
            self.cmd_vel_pub.publish(self.msg_)
            self.experiment_complete = True
            return
        else:
            u_val = self.u(self.settings['x_d'][0, :, self.index], self.settings['v_d'][0, :, self.index], self.index * self.settings['dt'])

            # Publish velocity command
            self.msg_.linear.x = float(u_val[0]) * 0.5
            self.msg_.linear.y = float(u_val[1]) * 0.5
            self.msg_.linear.z = 0.0
            self.msg_.angular.x = 0.0
            self.msg_.angular.y = 0.0
            self.msg_.angular.z = float(u_val[2]) * 0.5
            self.cmd_vel_pub.publish(self.msg_)
        self.index += 1
        
    def mocap_callback(self, msg):
        """
        Unified callback for all mocap position updates.
        Routes updates based on subject_name.
        """
        subject = msg.subject_name.lower()
        
        # Extract 2D position
        position = 0.001 * np.array([[msg.x_trans], [msg.y_trans]])     # raw data is in [mm]
        position = np.dot(self.mocap_R, position)

        if (self.mocap_P is None) and ('go_1' in subject):
            self.mocap_P = position
        else:
            position = position - self.mocap_P
            if 'go_1' in subject:
                quat = [msg.x_rot, msg.y_rot, msg.z_rot, msg.w]
                try:
                    rotation = Rotation.from_quat(quat)
                except ValueError as e:
                    self.get_logger().error(f"Invalid quaternion data: {quat}. Error: {e}")
                    return
                euler = rotation.as_euler('xyz', degrees=False)
                self.yaw = euler[2] + self.theta
                print(f"Robot Heading: {self.yaw * 180 / 3.1415}")
                # Update robot position
                self.x = np.vstack((position, self.yaw)) 
                self.position_received = True
                self.get_logger().debug(f"Robot position: {position.flatten()}")
                
            elif 'box_1' in subject:
                # Update obstacle 1 position
                self.obs1_position = position
                self.settings['obs1']['center'] = position
                self.get_logger().debug(f"Obstacle 1 position: {position.flatten()}")
                
            elif 'box_2' in subject:
                # Update obstacle 2 position
                self.obs2_position = position
                self.settings['obs2']['center'] = position
                self.get_logger().debug(f"Obstacle 2 position: {position.flatten()}")
                
            else:
                self.get_logger().error(f"Unknown subject: {msg.subject_name}")

def dynamics(x, u):
    """ Omni-directional robot dynamics without an unknown constant drift 'd'. """
    xdot = np.array([[(u[0]*np.cos(x[2]) - u[1]*np.sin(x[2])).item()], 
                     [(u[0]*np.sin(x[2]) + u[1]*np.cos(x[2])).item()], 
                     [u[2]]])
    drift = np.array([[0.2], [-0.1], [0.5]])
    return xdot + drift

def r4k(x, u, dt, func):
    """
    4th-order Runge-Kutta integration for continuous-time simulation.
    
    Args:
        x (np.ndarray): Current state vector.
        u (float): System input.
        dt (float): Time step.
        func (callable): The system dynamics function (e.g., damper).
        param (object): Parameters for the dynamics function.
        
    Returns:
        np.ndarray: The next state vector.
    """
    # RK4 algorithm from r4k.m
    a = func(x, u)
    b = func(x + 0.5 * a * dt, u)
    c = func(x + 0.5 * b * dt, u)
    d = func(x + c * dt, u)
    
    x_next = x + (dt / 6.0) * (a + 2*b + 2*c + d)
    return x_next

def run_simulation(args=None):
    rclpy.init(args=args)
    # --- Settings and Parameters --- 
    num_rbf_centers_1d = 5
    n_rbf_1d = num_rbf_centers_1d**3 + 1   
    settings = {
        'dt': 0.01, 't_end': 30, 'n_x': 3, 'n_u': 3, 'n_p': 3 * n_rbf_1d, 
        'kappa': 10.0, 'u_max': np.array([2.5, 2.5, 5.0]), 'w_max_norm': 1.0, 'E': 0.1,
        'obs1': {'center': np.array([[1.5], [-1.00]]), 'radius': 0.45},
        'obs2': {'center': np.array([[2.5], [-2.50]]), 'radius': 0.45},
        'workspace': [-1, 5, 1, -4, -np.pi, np.pi] # x_min, x_max, y_min, y_max theta_min, theta_max for RBF centers
    }
    settings['time'] = np.arange(0, settings['t_end'], settings['dt'])

    
    goal_pos = np.array([[4.0], [-3.0], [np.arctan2(-3.0, 4.0)]])
    settings['x_d'] = np.linspace(np.array([[0.0], [0.0], [0.0]]), goal_pos, len(settings['time'])).T
    settings['v_d'] = np.gradient(settings['x_d'], settings['dt'], axis=2)

    controller_params = {
        'K': np.diag([1.0, 1.0, 0.1]), 'gamma': np.diag([0.1] * settings['n_p']),
        'K_e': 0.025, 'w_max': np.ones((settings['n_p'], 1)) * settings['w_max_norm'],
        'u_max': settings['u_max'], 
        'alpha': 2.0, 'E': settings['E'], 'n_x': settings['n_x'], 
        'n_p': settings['n_p'], 'dt': settings['dt']
    }

    ctrl = MobileRobotController(settings=settings, **controller_params)
    start_time = time.time()
    x = np.array([[0.0], [0.0], [0.0]]) # Initial position
    
    for k, t in enumerate(settings['time']):
        ctrl.x = x
        u_val = ctrl.u(settings['x_d'][0, :, k], settings['v_d'][0, :, k], t)
        x = r4k(x, u_val, settings['dt'], dynamics)

    end_time = time.time()
    print(f"Execution time: {(end_time - start_time):.4f} seconds")
    plot_results(settings, ctrl, goal_pos)
    return

def run_experiment(args=None):
    rclpy.init(args=args)
    # --- Settings and Parameters --- 
    num_rbf_centers_1d = 5
    n_rbf_1d = num_rbf_centers_1d**3 + 1   
    settings = {
        'dt': 0.01, 't_end': 45, 'n_x': 3, 'n_u': 3, 'n_p': 3 * n_rbf_1d, 
        'kappa': 10.0, 'u_max': np.array([0.6, 0.3, 0.5]), 'w_max_norm': 1.0, 'E': 0.1,
        'obs1': {'center': np.array([[1.5], [-1.00]]), 'radius': 0.45},
        'obs2': {'center': np.array([[2.5], [-2.50]]), 'radius': 0.45},
        'workspace': [-1, 5, 1, -4, -np.pi, np.pi] # x_min, x_max, y_min, y_max theta_min, theta_max for RBF centers
    }
    settings['time'] = np.arange(0, settings['t_end'], settings['dt'])

    
    goal_pos = np.array([[4.0], [-3.0], [np.arctan2(-3.0, 4.0)]])
    settings['x_d'] = np.linspace(np.array([[0.0], [0.0], [0.0]]), goal_pos, len(settings['time'])).T
    settings['v_d'] = np.gradient(settings['x_d'], settings['dt'], axis=2)

    controller_params = {
        'K': np.diag([1.0, 1.0, 0.1]), 'gamma': np.diag([0.1] * settings['n_p']),
        'K_e': 0.01, 'w_max': np.ones((settings['n_p'], 1)) * settings['w_max_norm'],
        'u_max': settings['u_max'], 
        'alpha': 1.0, 'E': settings['E'], 'n_x': settings['n_x'], 
        'n_p': settings['n_p'], 'dt': settings['dt']
    }

    ctrl = MobileRobotController(settings=settings, **controller_params)
    
    try:
        while rclpy.ok() and not ctrl.experiment_complete:
            rclpy.spin_once(ctrl)
    except KeyboardInterrupt:
        ctrl.get_logger().info('KeyboardInterrupt received, shutting down.')
    finally:
        plot_results(settings, ctrl, goal_pos)
        ctrl.destroy_node()


def plot_results(settings, ctrl, goal_pos):
    """
    Create two publication-quality figures:
    1. Robot trajectory with obstacles
    2. Subplots showing: state trajectory, CBF history, slack variables, parameter estimates, control inputs
    
    Saves figures with timestamps and stores data in .npz files for later retrieval.
    """
    from datetime import datetime
    import os
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create fig directory if it doesn't exist
    os.makedirs('fig', exist_ok=True)
    
    # Prepare data
    time = np.arange(0, settings['t_end'], settings['dt'])
    x_hist = np.array(ctrl.joint_traj).squeeze()
    u_hist = np.array(ctrl.u_traj).squeeze()
    w_hist = np.array(ctrl.w_traj).squeeze()
    cbf_hist = np.array(ctrl.cbf_traj).squeeze()
    slack_hist = np.array(ctrl.slack_traj).squeeze()
    
    # Save data to npz file
    data_filename = f'fig/experiment_data_{timestamp}.npz'
    np.savez(data_filename,
             time=time,
             x_hist=x_hist,
             u_hist=u_hist,
             w_hist=w_hist,
             cbf_hist=cbf_hist,
             slack_hist=slack_hist,
             x_desired=settings['x_d'],
             v_desired=settings['v_d'],
             goal_pos=goal_pos,
             obs1_center=settings['obs1']['center'],
             obs1_radius=settings['obs1']['radius'],
             obs2_center=settings['obs2']['center'],
             obs2_radius=settings['obs2']['radius'],
             u_max=settings['u_max'],
             w_max_norm=settings['w_max_norm'],
             workspace=settings['workspace'],
             dt=settings['dt'],
             t_end=settings['t_end'])
    print(f"Saved data: {data_filename}")
    
    # Define color palette (colorblind-friendly)
    colors = {
        'trajectory': '#2E86AB',  # Blue
        'start': '#06A77D',       # Green
        'end': '#D62828',         # Red
        'goal': '#F77F00',        # Orange
        'obstacle': '#EF233C',    # Red
        'reference': '#6C757D',   # Gray
        'u_x': '#2E86AB',         # Blue
        'u_y': '#D62828',         # Red
        'u_theta': '#F77F00',     # Orange
        'cbf': '#06A77D',         # Green
        'param': '#06A77D',       # Green
        'slack_clf': '#2E86AB',   # Blue
        'slack_cbf': '#D62828',   # Red
    }
    
    # ========== Figure 1: Trajectory and Obstacles ==========
    fig1 = plt.figure(figsize=(5, 4))
    ax1 = fig1.add_subplot(111)
    
    # Plot robot trajectory with gradient effect
    ax1.plot(x_hist[:, 0], x_hist[:, 1], color=colors['trajectory'], 
             linewidth=2, label='Robot trajectory', zorder=3)
    
    # Mark start and end positions
    ax1.plot(x_hist[0, 0], x_hist[0, 1], 'o', color=colors['start'], 
             markersize=10, label='Start', markeredgecolor='black', 
             markeredgewidth=1.0, zorder=5)
    ax1.plot(x_hist[-1, 0], x_hist[-1, 1], 's', color=colors['end'], 
             markersize=10, label='End', markeredgecolor='black', 
             markeredgewidth=1.0, zorder=5)
    
    # Plot obstacles with better styling
    obs1_circle = Circle(settings['obs1']['center'].flatten(), 
                         settings['obs1']['radius'], 
                         facecolor=colors['obstacle'], alpha=0.6, 
                         edgecolor=colors['obstacle'], linewidth=2, 
                         label='Obstacles', zorder=2)
    obs2_circle = Circle(settings['obs2']['center'].flatten(), 
                         settings['obs2']['radius'], 
                         facecolor=colors['obstacle'], alpha=0.6,
                         edgecolor=colors['obstacle'], linewidth=2, zorder=2)
    ax1.add_patch(obs1_circle)
    ax1.add_patch(obs2_circle)
    
    # Plot goal position
    ax1.plot(goal_pos[0], goal_pos[1], '*', color=colors['goal'], 
             markersize=18, label='Goal', markeredgecolor='black', 
             markeredgewidth=1.2, zorder=5, alpha=0.5)
    
    # Set equal aspect ratio and labels
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel('$x$ [m]', fontsize=11)
    ax1.set_ylabel('$y$ [m]', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.95, edgecolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_xlim([settings['workspace'][0] - 0.5, settings['workspace'][1] + 0.5])
    ax1.set_ylim([settings['workspace'][2] - 0.5, settings['workspace'][3] - 0.5])
    
    # Add minor ticks
    ax1.minorticks_on()
    ax1.tick_params(which='both', direction='in', top=True, right=True)
    
    plt.tight_layout()
    fig1_filename = f'fig/mobile_robot_path_{timestamp}.pdf'
    plt.savefig(fig1_filename, format='pdf', bbox_inches='tight')
    print(f"Saved: {fig1_filename}")

    # ========== Figure 2: History Subplots (3x2 layout) ==========
    fig2, axs = plt.subplots(3, 2, figsize=(7.5, 8))
    
    # [0,0] State Trajectory - X position
    axs[0, 0].plot(time, x_hist[:, 0], color=colors['trajectory'], 
                   linewidth=1.5, label='$x(t)$')
    axs[0, 0].plot(time, settings['x_d'][0, 0, :], '--', 
                   color=colors['reference'], linewidth=1.5, label='$x_\\mathrm{goal}$')
    axs[0, 0].set_ylabel('$x$ [m]', fontsize=11)
    axs[0, 0].legend(loc='best', fontsize=9, framealpha=0.95)
    axs[0, 0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axs[0, 0].tick_params(which='both', direction='in', top=True, right=True)
    axs[0, 0].set_xticklabels([])
    
    # [0,1] State Trajectory - Y position
    axs[0, 1].plot(time, x_hist[:, 1], color=colors['trajectory'], 
                   linewidth=1.5, label='$y(t)$')
    axs[0, 1].plot(time, settings['x_d'][0, 1, :], '--', 
                   color=colors['reference'], linewidth=1.5, label='$y_\\mathrm{goal}$')
    axs[0, 1].set_ylabel('$y$ [m]', fontsize=11)
    axs[0, 1].legend(loc='best', fontsize=9, framealpha=0.95)
    axs[0, 1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axs[0, 1].tick_params(which='both', direction='in', top=True, right=True)
    axs[0, 1].set_xticklabels([])
    
    # [1,0] CBF Value
    axs[1, 0].plot(time, cbf_hist, color=colors['trajectory'], 
                   linewidth=1.5, label='$h(x)$')
    axs[1, 0].axhline(y=0, color=colors['end'], linestyle='--', 
                      linewidth=1.5, label='Safety boundary')
    axs[1, 0].set_ylabel('$h(x)$', fontsize=11)
    axs[1, 0].legend(loc='best', fontsize=9, framealpha=0.95)
    axs[1, 0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axs[1, 0].tick_params(which='both', direction='in', top=True, right=True)
    axs[1, 0].set_xticklabels([])
    
    # [1,1] Control Input
    axs[1, 1].plot(time, u_hist[:, 0], color=colors['u_x'], 
                   linewidth=1.5, label='$u_x$')
    axs[1, 1].plot(time, u_hist[:, 1], color=colors['u_y'], 
                   linewidth=1.5, label='$u_y$')
    axs[1, 1].plot(time, u_hist[:, 2], color=colors['u_theta'], 
                   linewidth=1.5, label='$u_\\theta$')
    axs[1, 1].axhline(y=settings['u_max'][0], color=colors['u_x'], linestyle=':', 
                      linewidth=1.2, label='Input limits', alpha=0.7)
    axs[1, 1].axhline(y=-settings['u_max'][0], color=colors['u_x'], linestyle=':', 
                      linewidth=1.2, alpha=0.7)
    axs[1, 1].axhline(y=settings['u_max'][2], color=colors['u_theta'], linestyle=':', 
                      linewidth=1.2, alpha=0.7)
    axs[1, 1].set_ylabel('Control [m/s]', fontsize=11)
    axs[1, 1].legend(loc='best', fontsize=9, framealpha=0.95)
    axs[1, 1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axs[1, 1].tick_params(which='both', direction='in', top=True, right=True)
    axs[1, 1].set_xticklabels([])
    
    # [2,0] Parameter Estimation (norm)
    w_norm = np.linalg.norm(w_hist, axis=1)
    axs[2, 0].plot(time, w_norm, color=colors['param'], linewidth=1.5, 
                   label='$\\|\\hat{w}\\|$')
    axs[2, 0].axhline(y=settings['w_max_norm'], color=colors['end'], 
                      linestyle='--', linewidth=1.5, 
                      label=f'Upper bound', alpha=0.8)
    axs[2, 0].set_ylabel('$\\|\\hat{w}\\|$', fontsize=11)
    axs[2, 0].set_xlabel('Time [s]', fontsize=11)
    axs[2, 0].legend(loc='best', fontsize=9, framealpha=0.95)
    axs[2, 0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axs[2, 0].tick_params(which='both', direction='in', top=True, right=True)
    
    # [2,1] Slack Variables
    axs[2, 1].plot(time, slack_hist[:, 0], color=colors['slack_clf'], 
                   linewidth=1.5, label='CLF slack')
    axs[2, 1].plot(time, slack_hist[:, 1], color=colors['slack_cbf'], 
                   linewidth=1.5, label='CBF slack')
    axs[2, 1].set_ylabel('Slack variables', fontsize=11)
    axs[2, 1].set_xlabel('Time [s]', fontsize=11)
    # Remove the log scale line
    axs[2, 1].legend(loc='best', fontsize=9, framealpha=0.95)
    axs[2, 1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axs[2, 1].tick_params(which='both', direction='in', top=True, right=True)
    
    plt.tight_layout()
    fig2_filename = f'fig/mobile_robot_results_{timestamp}.pdf'
    plt.savefig(fig2_filename, format='pdf', bbox_inches='tight')
    print(f"Saved: {fig2_filename}")
    # plt.show()

if __name__ == '__main__':
    run_experiment()
    # run_simulation()