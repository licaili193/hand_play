"""
Standalone Hand Gesture Visualization Script

This script visualizes predicted hand gestures from a comprehensive JSON file containing
frame-by-frame hand data including positions, angles, joint coordinates, and bone lengths.
It is completely self-contained and only depends on standard plotting libraries.

Features:
- 3D visualization of both hands with anatomically correct finger coloring
- Interactive frame navigation with slider control
- Animation playback with adjustable speed
- Export capabilities for individual frames or animations
- Real-time statistics and motion analysis
- Standalone operation - no external dependencies beyond plotting libraries
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
import argparse
import os
from typing import List, Dict, Tuple, Optional
import time

class StandaloneHandGestureVisualizer:
    """
    A completely standalone hand gesture visualizer for piano motion data.
    Only depends on standard plotting libraries.
    """
    
    def __init__(self, json_file_path: str, view_mode: str = '3d'):
        """
        Initialize the visualizer with comprehensive hand motion data.
        
        Args:
            json_file_path (str): Path to the comprehensive JSON file containing hand motion data
            view_mode (str): Visualization mode - '3d' or '2d_topdown'
        """
        self.json_file_path = json_file_path
        self.view_mode = view_mode.lower()
        if self.view_mode not in ['3d', '2d_topdown']:
            raise ValueError("view_mode must be '3d' or '2d_topdown'")
            
        self.data = self.load_data()
        self.frames = self.data.get('frames', [])
        self.num_frames = len(self.frames)
        
        # Extract metadata
        self.metadata = self.data.get('metadata', {})
        self.mano_joint_names = self.metadata.get('mano_joint_names', [])
        self.mano_joint_connections = self.metadata.get('mano_joint_connections', [])
        self.finger_groups = self.metadata.get('finger_groups', {})
        
        # Animation state
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0  # frames per second
        
        # Visualization settings
        self.show_labels = True
        self.show_connections = True
        self.auto_scale = True
        
        print(f"Loaded {self.num_frames} frames from {json_file_path}")
        print(f"Data format: {self.metadata.get('data_format', 'unknown')}")
        print(f"Joints per hand: {self.metadata.get('joints_per_hand', 'unknown')}")
        print(f"View mode: {self.view_mode.upper()}")
    
    def load_data(self) -> Dict:
        """
        Load comprehensive hand motion data from JSON file.
        
        Returns:
            Dict: Loaded data containing frames and metadata
        """
        try:
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {self.json_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {self.json_file_path}")
    
    def get_frame_data(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        Extract comprehensive hand data for a given frame.
        
        Args:
            frame_idx (int): Frame index
            
        Returns:
            Tuple containing:
            - left_hand_joints: Left hand joint coordinates (21, 3)
            - right_hand_joints: Right hand joint coordinates (21, 3)
            - left_hand_info: Left hand additional info (position, angles, bone lengths)
            - right_hand_info: Right hand additional info (position, angles, bone lengths)
        """
        if frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range (0-{self.num_frames-1})")
        
        frame_data = self.frames[frame_idx]
        
        # Extract joint coordinates (already calculated)
        left_hand_joints = np.array(frame_data['left_hand_joints'])
        right_hand_joints = np.array(frame_data['right_hand_joints'])
        
        # Extract additional information
        left_hand_info = {
            'position': np.array(frame_data['left_hand_position']),
            'angles': np.array(frame_data['left_hand_angles']),
            'bone_lengths': np.array(frame_data['left_hand_bone_lengths'])
        }
        
        right_hand_info = {
            'position': np.array(frame_data['right_hand_position']),
            'angles': np.array(frame_data['right_hand_angles']),
            'bone_lengths': np.array(frame_data['right_hand_bone_lengths'])
        }
        
        return left_hand_joints, right_hand_joints, left_hand_info, right_hand_info
    
    def calculate_motion_statistics(self) -> Dict:
        """
        Calculate motion statistics across all frames.
        
        Returns:
            Dict: Motion statistics
        """
        if self.num_frames < 2:
            return {"error": "Need at least 2 frames for motion analysis"}
        
        # Calculate joint velocities (movement between frames)
        velocities = []
        for i in range(1, self.num_frames):
            left_prev, right_prev, _, _ = self.get_frame_data(i-1)
            left_curr, right_curr, _, _ = self.get_frame_data(i)
            
            # Calculate velocity for each joint
            left_vel = np.linalg.norm(left_curr - left_prev, axis=1)
            right_vel = np.linalg.norm(right_curr - right_prev, axis=1)
            
            velocities.append({
                'frame': i,
                'left_hand_velocities': left_vel.tolist(),
                'right_hand_velocities': right_vel.tolist(),
                'left_hand_avg_velocity': np.mean(left_vel),
                'right_hand_avg_velocity': np.mean(right_vel)
            })
        
        # Calculate overall statistics
        left_avg_velocities = [v['left_hand_avg_velocity'] for v in velocities]
        right_avg_velocities = [v['right_hand_avg_velocity'] for v in velocities]
        
        stats = {
            'total_frames': self.num_frames,
            'duration_seconds': self.num_frames / 30.0,  # Assuming 30 FPS
            'left_hand_stats': {
                'max_velocity': max(left_avg_velocities),
                'min_velocity': min(left_avg_velocities),
                'mean_velocity': np.mean(left_avg_velocities),
                'std_velocity': np.std(left_avg_velocities)
            },
            'right_hand_stats': {
                'max_velocity': max(right_avg_velocities),
                'min_velocity': min(right_avg_velocities),
                'mean_velocity': np.mean(right_avg_velocities),
                'std_velocity': np.std(right_avg_velocities)
            },
            'frame_by_frame_velocities': velocities
        }
        
        return stats
    
    def plot_hand_3d(self, joints: np.ndarray, connections: List[Tuple[int, int]], 
                    title: str, color: str, ax=None, alpha: float = 1.0, 
                    add_to_existing: bool = False):
        """
        Plot a hand in 3D using joint coordinates and connections.
        
        Args:
            joints: Joint coordinates (21, 3)
            connections: List of (parent, child) joint connections
            title: Title for the plot
            color: Color for the hand
            ax: Matplotlib 3D axis
            alpha: Transparency
            add_to_existing: Whether to add to existing plot
        """
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
        
        if not add_to_existing:
            ax.clear()
        
        # Plot joints
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                  c=color, s=50, alpha=alpha, label=title)
        
        # Plot connections
        for parent, child in connections:
            if parent < len(joints) and child < len(joints):
                ax.plot([joints[parent, 0], joints[child, 0]],
                       [joints[parent, 1], joints[child, 1]],
                       [joints[parent, 2], joints[child, 2]], 
                       color=color, linewidth=2, alpha=alpha)
        
        # Highlight key joints (wrist and fingertips)
        key_joints = [0, 4, 8, 12, 16, 20]  # Wrist and fingertips
        for joint_idx in key_joints:
            if joint_idx < len(joints):
                ax.scatter(joints[joint_idx, 0], joints[joint_idx, 1], joints[joint_idx, 2], 
                          c='black', s=80, alpha=alpha*0.8, marker='o', edgecolors=color)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        
        # Set axis limits (same as midi_to_frames.py)
        max_range = 0.1  # Adjust based on hand size
        ax.set_xlim([joints[0, 0] - max_range, joints[0, 0] + max_range])
        ax.set_ylim([joints[0, 1] - max_range, joints[0, 1] + max_range])
        ax.set_zlim([joints[0, 2] - max_range, joints[0, 2] + max_range])
        
        return ax
    
    def plot_hand_2d_topdown(self, joints: np.ndarray, connections: List[Tuple[int, int]], 
                           title: str, color: str, ax=None, show_labels: bool = True, add_to_existing: bool = False):
        """
        Plot a hand in 2D top-down view using joint coordinates and connections.
        
        Args:
            joints: Joint coordinates (21, 3)
            connections: List of (parent, child) joint connections
            title: Title for the plot
            color: Color for the hand
            ax: Matplotlib 2D axis
            show_labels: Whether to show joint labels
            add_to_existing: Whether to add to existing plot
        """
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
        
        if not add_to_existing:
            ax.clear()
        
        # Plot joints
        ax.scatter(joints[:, 0], joints[:, 1], c=color, s=50, alpha=0.8, 
                  edgecolors='black', linewidth=1, label=title)
        
        # Plot connections
        for parent, child in connections:
            if parent < len(joints) and child < len(joints):
                ax.plot([joints[parent, 0], joints[child, 0]],
                       [joints[parent, 1], joints[child, 1]], 
                       color=color, linewidth=2, alpha=0.8)
        
        # Highlight key joints
        key_joints = [0, 4, 8, 12, 16, 20]  # Wrist and fingertips
        for joint_idx in key_joints:
            if joint_idx < len(joints):
                ax.scatter(joints[joint_idx, 0], joints[joint_idx, 1], 
                          c='black', s=80, alpha=0.8, marker='o', edgecolors=color)
        
        if show_labels and self.mano_joint_names:
            # Add labels for key joints
            key_labels = {0: 'Wrist', 4: 'Thumb', 8: 'Index', 12: 'Middle', 16: 'Ring', 20: 'Little'}
            for joint_idx, label in key_labels.items():
                if joint_idx < len(joints):
                    ax.text(joints[joint_idx, 0], joints[joint_idx, 1], 
                           label, fontsize=10, fontweight='bold', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        return ax
    
    def visualize_single_frame(self, frame_idx: int, save_path: Optional[str] = None):
        """
        Visualize a single frame with both hands on the same plot.
        
        Args:
            frame_idx (int): Frame index to visualize
            save_path (str, optional): Path to save the visualization
        """
        left_hand, right_hand, left_info, right_info = self.get_frame_data(frame_idx)
        
        fig = plt.figure(figsize=(12, 10))
        
        if self.view_mode == '3d':
            # Create single 3D plot for both hands
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot both hands on the same plot
            self.plot_hand_3d(left_hand, self.mano_joint_connections, 
                             f"Left Hand - Frame {frame_idx}", 'blue', ax, alpha=0.8)
            self.plot_hand_3d(right_hand, self.mano_joint_connections, 
                             f"Right Hand - Frame {frame_idx}", 'red', ax, alpha=0.8, add_to_existing=True)
            
            # Set consistent view angle
            ax.view_init(elev=20, azim=45)
            
            # Set consistent axis limits based on both hands
            all_joints = np.vstack([left_hand, right_hand])
            center = np.mean(all_joints, axis=0)
            max_range = np.max(np.ptp(all_joints, axis=0)) * 0.6  # 60% of the range
            
            ax.set_xlim([center[0] - max_range, center[0] + max_range])
            ax.set_ylim([center[1] - max_range, center[1] + max_range])
            ax.set_zlim([center[2] - max_range, center[2] + max_range])
        
        else:  # 2D top-down view
            # Create single 2D plot for both hands
            ax = fig.add_subplot(111)
            
            # Plot both hands in 2D top-down view
            self.plot_hand_2d_topdown(left_hand, self.mano_joint_connections, 
                                    f"Left Hand - Frame {frame_idx}", 'blue', ax, self.show_labels)
            self.plot_hand_2d_topdown(right_hand, self.mano_joint_connections, 
                                    f"Right Hand - Frame {frame_idx}", 'red', ax, self.show_labels, add_to_existing=True)
            
            # Set consistent scaling for both hands
            if self.auto_scale:
                all_coords = np.vstack([left_hand, right_hand])
                x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
                y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
                
                # Add some padding
                padding = 0.1
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
                ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
        
        view_mode_text = "3D" if self.view_mode == '3d' else "2D Top-Down"
        plt.title(f'Hand Gesture Visualization - Frame {frame_idx} of {self.num_frames-1} ({view_mode_text})', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def create_interactive_visualizer(self):
        """
        Create an interactive visualization with controls for frame navigation and playback.
        """
        fig = plt.figure(figsize=(14, 10))
        
        # Create main visualization area (single plot for both hands)
        if self.view_mode == '3d':
            ax = fig.add_subplot(2, 3, (1, 2), projection='3d')
        else:
            ax = fig.add_subplot(2, 3, (1, 2))
        
        ax_stats = fig.add_subplot(2, 3, 3)
        
        # Create control area
        ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
        ax_play = fig.add_axes([0.1, 0.15, 0.1, 0.04])
        ax_speed = fig.add_axes([0.25, 0.15, 0.15, 0.04])
        ax_export = fig.add_axes([0.45, 0.15, 0.1, 0.04])
        ax_stats_btn = fig.add_axes([0.6, 0.15, 0.1, 0.04])
        
        # Initialize visualization
        left_hand, right_hand, left_info, right_info = self.get_frame_data(0)
        
        # Create initial plot with both hands
        if self.view_mode == '3d':
            self.plot_hand_3d(left_hand, self.mano_joint_connections, "Left Hand", 'blue', ax, alpha=0.8)
            self.plot_hand_3d(right_hand, self.mano_joint_connections, "Right Hand", 'red', ax, alpha=0.8, add_to_existing=True)
            
            # Set consistent view angle
            ax.view_init(elev=20, azim=45)
            
            # Set consistent axis limits based on both hands
            all_joints = np.vstack([left_hand, right_hand])
            center = np.mean(all_joints, axis=0)
            max_range = np.max(np.ptp(all_joints, axis=0)) * 0.6
            
            ax.set_xlim([center[0] - max_range, center[0] + max_range])
            ax.set_ylim([center[1] - max_range, center[1] + max_range])
            ax.set_zlim([center[2] - max_range, center[2] + max_range])
        else:
            self.plot_hand_2d_topdown(left_hand, self.mano_joint_connections, "Left Hand", 'blue', ax, self.show_labels)
            self.plot_hand_2d_topdown(right_hand, self.mano_joint_connections, "Right Hand", 'red', ax, self.show_labels, add_to_existing=True)
            
            # Set consistent scaling for both hands
            if self.auto_scale:
                all_coords = np.vstack([left_hand, right_hand])
                x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
                y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
                
                padding = 0.1
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
                ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
        
        # Create slider
        slider = Slider(ax_slider, 'Frame', 0, self.num_frames-1, valinit=0, valstep=1)
        
        # Create buttons
        play_button = Button(ax_play, 'Play/Pause')
        speed_button = Button(ax_speed, 'Speed: 1x')
        export_button = Button(ax_export, 'Export')
        stats_button = Button(ax_stats_btn, 'Stats')
        
        # Statistics text
        stats_text = ax_stats.text(0.1, 0.9, "Click 'Stats' to view motion statistics", 
                                  transform=ax_stats.transAxes, fontsize=10,
                                  verticalalignment='top')
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.axis('off')
        
        def update_frame(val):
            """Update visualization for new frame."""
            frame_idx = int(slider.val)
            left_hand, right_hand, left_info, right_info = self.get_frame_data(frame_idx)
            
            # Clear previous plots
            ax.clear()
            
            # Redraw both hands based on view mode
            if self.view_mode == '3d':
                self.plot_hand_3d(left_hand, self.mano_joint_connections, 
                                f"Left Hand - Frame {frame_idx}", 'blue', ax, alpha=0.8)
                self.plot_hand_3d(right_hand, self.mano_joint_connections, 
                                f"Right Hand - Frame {frame_idx}", 'red', ax, alpha=0.8, add_to_existing=True)
                
                # Set consistent view angle
                ax.view_init(elev=20, azim=45)
                
                # Set consistent axis limits based on both hands
                all_joints = np.vstack([left_hand, right_hand])
                center = np.mean(all_joints, axis=0)
                max_range = np.max(np.ptp(all_joints, axis=0)) * 0.6
                
                ax.set_xlim([center[0] - max_range, center[0] + max_range])
                ax.set_ylim([center[1] - max_range, center[1] + max_range])
                ax.set_zlim([center[2] - max_range, center[2] + max_range])
            else:
                self.plot_hand_2d_topdown(left_hand, self.mano_joint_connections, 
                                        f"Left Hand - Frame {frame_idx}", 'blue', ax, self.show_labels)
                self.plot_hand_2d_topdown(right_hand, self.mano_joint_connections, 
                                        f"Right Hand - Frame {frame_idx}", 'red', ax, self.show_labels, add_to_existing=True)
                
                # Set consistent scaling for both hands
                if self.auto_scale:
                    all_coords = np.vstack([left_hand, right_hand])
                    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
                    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
                    
                    padding = 0.1
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    
                    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
                    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
            
            fig.canvas.draw_idle()
        
        def play_animation(event):
            """Toggle animation playback."""
            nonlocal self
            self.is_playing = not self.is_playing
            if self.is_playing:
                animate()
        
        def animate():
            """Animate through frames."""
            if self.is_playing:
                current_val = slider.val
                next_val = current_val + self.playback_speed
                if next_val >= self.num_frames - 1:
                    next_val = 0  # Loop back to start
                slider.set_val(next_val)
                plt.pause(0.1)  # 100ms delay
                plt.gcf().canvas.draw()
                plt.gcf().canvas.flush_events()
        
        def change_speed(event):
            """Change playback speed."""
            nonlocal self
            speeds = [0.5, 1.0, 2.0, 5.0]
            current_idx = speeds.index(self.playback_speed)
            next_idx = (current_idx + 1) % len(speeds)
            self.playback_speed = speeds[next_idx]
            speed_button.label.set_text(f'Speed: {self.playback_speed}x')
        
        def export_frame(event):
            """Export current frame as image."""
            frame_idx = int(slider.val)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"hand_gesture_frame_{frame_idx:04d}_{timestamp}.png"
            self.visualize_single_frame(frame_idx, filename)
        
        def show_stats(event):
            """Show motion statistics."""
            stats = self.calculate_motion_statistics()
            
            # Clear stats area
            ax_stats.clear()
            ax_stats.axis('off')
            
            # Display statistics
            stats_text = f"""Motion Statistics:
            
Total Frames: {stats['total_frames']}
Duration: {stats['duration_seconds']:.2f} seconds

Left Hand:
  Max Velocity: {stats['left_hand_stats']['max_velocity']:.4f}
  Mean Velocity: {stats['left_hand_stats']['mean_velocity']:.4f}
  Std Velocity: {stats['left_hand_stats']['std_velocity']:.4f}

Right Hand:
  Max Velocity: {stats['right_hand_stats']['max_velocity']:.4f}
  Mean Velocity: {stats['right_hand_stats']['mean_velocity']:.4f}
  Std Velocity: {stats['right_hand_stats']['std_velocity']:.4f}"""
            
            ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                         fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        # Connect events
        slider.on_changed(update_frame)
        play_button.on_clicked(play_animation)
        speed_button.on_clicked(change_speed)
        export_button.on_clicked(export_frame)
        stats_button.on_clicked(show_stats)
        
        plt.suptitle('Standalone Interactive Hand Gesture Visualizer', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def create_animation(self, output_path: str = "hand_gesture_animation.gif", 
                        fps: int = 10, dpi: int = 100):
        """
        Create an animated GIF of the hand gestures with consistent axis ranges.
        
        Args:
            output_path (str): Path to save the animation
            fps (int): Frames per second for the animation
            dpi (int): DPI for the output animation
        """
        fig = plt.figure(figsize=(12, 10))
        
        # Create single plot for both hands
        if self.view_mode == '3d':
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        
        # Calculate global axis limits across all frames for consistency
        print("Calculating global axis limits for consistent animation...")
        all_joints = []
        for i in range(self.num_frames):
            left_hand, right_hand, _, _ = self.get_frame_data(i)
            all_joints.append(left_hand)
            all_joints.append(right_hand)
        
        all_joints = np.vstack(all_joints)
        
        if self.view_mode == '3d':
            # Calculate 3D global limits
            center = np.mean(all_joints, axis=0)
            max_range = np.max(np.ptp(all_joints, axis=0)) * 0.6
            
            global_xlim = [center[0] - max_range, center[0] + max_range]
            global_ylim = [center[1] - max_range, center[1] + max_range]
            global_zlim = [center[2] - max_range, center[2] + max_range]
        else:
            # Calculate 2D global limits
            x_min, x_max = all_joints[:, 0].min(), all_joints[:, 0].max()
            y_min, y_max = all_joints[:, 1].min(), all_joints[:, 1].max()
            
            padding = 0.1
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            global_xlim = [x_min - padding * x_range, x_max + padding * x_range]
            global_ylim = [y_min - padding * y_range, y_max + padding * y_range]
        
        def animate(frame_idx):
            """Animation function for each frame."""
            left_hand, right_hand, left_info, right_info = self.get_frame_data(frame_idx)
            
            # Clear previous plots
            ax.clear()
            
            # Visualize both hands based on view mode
            if self.view_mode == '3d':
                self.plot_hand_3d(left_hand, self.mano_joint_connections, 
                                f"Left Hand - Frame {frame_idx}", 'blue', ax, alpha=0.8)
                self.plot_hand_3d(right_hand, self.mano_joint_connections, 
                                f"Right Hand - Frame {frame_idx}", 'red', ax, alpha=0.8, add_to_existing=True)
                
                # Set consistent view angle
                ax.view_init(elev=20, azim=45)
                
                # Use global axis limits for consistency
                ax.set_xlim(global_xlim)
                ax.set_ylim(global_ylim)
                ax.set_zlim(global_zlim)
            else:
                self.plot_hand_2d_topdown(left_hand, self.mano_joint_connections, 
                                        f"Left Hand - Frame {frame_idx}", 'blue', ax, False)
                self.plot_hand_2d_topdown(right_hand, self.mano_joint_connections, 
                                        f"Right Hand - Frame {frame_idx}", 'red', ax, False, add_to_existing=True)
                
                # Use global axis limits for consistency
                ax.set_xlim(global_xlim)
                ax.set_ylim(global_ylim)
                ax.set_aspect('equal')
            
            view_mode_text = "3D" if self.view_mode == '3d' else "2D Top-Down"
            plt.title(f'Hand Gesture Animation - Frame {frame_idx} of {self.num_frames-1} ({view_mode_text})', 
                     fontsize=14, fontweight='bold')
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=self.num_frames, 
                                     interval=1000//fps, repeat=True)
        
        # Save animation
        print(f"Creating animation with {self.num_frames} frames at {fps} FPS...")
        print(f"Using consistent axis limits for smooth animation")
        anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
        print(f"Animation saved to {output_path}")
        
        plt.show()

def main():
    """Main function to run the standalone hand gesture visualizer."""
    parser = argparse.ArgumentParser(description='Standalone visualization of hand gestures from comprehensive JSON data')
    parser.add_argument('json_file', help='Path to the comprehensive JSON file containing hand motion data')
    parser.add_argument('--mode', choices=['single', 'interactive', 'animation'], 
                       default='interactive', help='Visualization mode')
    parser.add_argument('--view', choices=['3d', '2d_topdown'], 
                       default='3d', help='View mode: 3D or 2D top-down')
    parser.add_argument('--frame', type=int, default=0, help='Frame to visualize (for single mode)')
    parser.add_argument('--save', help='Path to save the visualization (for single mode)')
    parser.add_argument('--output', default='hand_gesture_animation.gif', 
                       help='Output path for animation (for animation mode)')
    parser.add_argument('--fps', type=int, default=10, help='FPS for animation')
    
    args = parser.parse_args()
    
    # Create visualizer with specified view mode
    visualizer = StandaloneHandGestureVisualizer(args.json_file, args.view)
    
    # Run appropriate mode
    if args.mode == 'single':
        visualizer.visualize_single_frame(args.frame, args.save)
    elif args.mode == 'interactive':
        visualizer.create_interactive_visualizer()
    elif args.mode == 'animation':
        visualizer.create_animation(args.output, args.fps)
    
    # Print basic statistics
    print("\n=== Basic Motion Statistics ===")
    stats = visualizer.calculate_motion_statistics()
    print(f"Total frames: {stats['total_frames']}")
    print(f"Duration: {stats['duration_seconds']:.2f} seconds")
    print(f"Left hand mean velocity: {stats['left_hand_stats']['mean_velocity']:.4f}")
    print(f"Right hand mean velocity: {stats['right_hand_stats']['mean_velocity']:.4f}")

if __name__ == "__main__":
    main() 