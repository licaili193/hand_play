"""
Hand Gesture Visualization Script

This script visualizes predicted hand gestures from a JSON file containing
frame-by-frame hand joint coordinates. It provides both static 3D visualization
and animated playback of the hand motion sequence.

Features:
- 3D visualization of both hands with anatomically correct finger coloring
- Interactive frame navigation with slider control
- Animation playback with adjustable speed
- Export capabilities for individual frames or animations
- Real-time statistics and motion analysis
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

# Import our utility functions
from mano_utils import (
    get_finger_colors, 
    get_finger_ranges, 
    visualize_hand_3d,
    get_mano_joint_names
)

class HandGestureVisualizer:
    """
    A comprehensive hand gesture visualizer for piano motion data.
    """
    
    def __init__(self, json_file_path: str, view_mode: str = '3d'):
        """
        Initialize the visualizer with hand motion data.
        
        Args:
            json_file_path (str): Path to the JSON file containing hand motion data
            view_mode (str): Visualization mode - '3d' or '2d_topdown'
        """
        self.json_file_path = json_file_path
        self.view_mode = view_mode.lower()
        if self.view_mode not in ['3d', '2d_topdown']:
            raise ValueError("view_mode must be '3d' or '2d_topdown'")
            
        self.data = self.load_data()
        self.frames = self.data.get('frames', [])
        self.num_frames = len(self.frames)
        
        # Get joint structure information
        self.finger_colors = get_finger_colors()
        self.finger_ranges = get_finger_ranges()
        self.mano_joint_names = get_mano_joint_names()
        
        # Animation state
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0  # frames per second
        
        # Visualization settings
        self.show_labels = True
        self.show_connections = True
        self.auto_scale = True
        
        print(f"Loaded {self.num_frames} frames from {json_file_path}")
        print(f"Each frame contains 16 joints per hand (32 total joints)")
        print(f"View mode: {self.view_mode.upper()}")
    
    def load_data(self) -> Dict:
        """
        Load hand motion data from JSON file.
        
        Returns:
            Dict: Loaded data containing frames
        """
        try:
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {self.json_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {self.json_file_path}")
    
    def get_frame_data(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract left and right hand joint coordinates for a given frame.
        
        Args:
            frame_idx (int): Frame index
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Left and right hand joint coordinates
        """
        if frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range (0-{self.num_frames-1})")
        
        frame_data = self.frames[frame_idx]
        left_hand = np.array(frame_data['left_hand_joints'])
        right_hand = np.array(frame_data['right_hand_joints'])
        
        return left_hand, right_hand
    
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
            left_prev, right_prev = self.get_frame_data(i-1)
            left_curr, right_curr = self.get_frame_data(i)
            
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
    
    def set_uniform_axes(self, ax):
        """
        Set uniform tick spacing and grid for 3D axes.
        
        Args:
            ax: matplotlib 3D axis object
        """
        # Get current axis limits
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        z_lim = ax.get_zlim()
        
        # Calculate the range for each axis
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]
        z_range = z_lim[1] - z_lim[0]
        
        # Find the maximum range to determine uniform tick spacing
        max_range = max(x_range, y_range, z_range)
        
        # Calculate number of ticks (aim for 5-8 ticks per axis)
        num_ticks = 6
        tick_spacing = max_range / num_ticks
        
        # Set uniform tick spacing for all axes
        ax.set_xticks(np.arange(x_lim[0], x_lim[1] + tick_spacing, tick_spacing))
        ax.set_yticks(np.arange(y_lim[0], y_lim[1] + tick_spacing, tick_spacing))
        ax.set_zticks(np.arange(z_lim[0], z_lim[1] + tick_spacing, tick_spacing))
        
        # Enable grid with uniform spacing
        ax.grid(True, alpha=0.3)
        
        # Set aspect ratio to be equal for all axes
        ax.set_box_aspect([1, 1, 1])
    
    def set_uniform_axes_2d(self, ax):
        """
        Set uniform tick spacing and grid for 2D axes.
        
        Args:
            ax: matplotlib 2D axis object
        """
        # Get current axis limits
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        
        # Calculate the range for each axis
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]
        
        # Find the maximum range to determine uniform tick spacing
        max_range = max(x_range, y_range)
        
        # Calculate number of ticks (aim for 5-8 ticks per axis)
        num_ticks = 6
        tick_spacing = max_range / num_ticks
        
        # Set uniform tick spacing for all axes
        ax.set_xticks(np.arange(x_lim[0], x_lim[1] + tick_spacing, tick_spacing))
        ax.set_yticks(np.arange(y_lim[0], y_lim[1] + tick_spacing, tick_spacing))
        
        # Enable grid with uniform spacing
        ax.grid(True, alpha=0.3)
        
        # Set aspect ratio to be equal
        ax.set_aspect('equal')
    
    def visualize_hand_2d_topdown(self, coords, title="Hand", ax=None, show_labels=True):
        """
        Visualize a single hand in 2D top-down view (X-Y plane).
        
        Args:
            coords (np.ndarray): Joint coordinates of shape (num_joints, 3)
            title (str): Title for the plot
            ax (matplotlib.axes.Axes, optional): Existing 2D axis to plot on
            show_labels (bool): Whether to show joint labels
            
        Returns:
            matplotlib.axes.Axes: The axis object
        """
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
        
        finger_colors = self.finger_colors
        finger_ranges = self.finger_ranges
        
        # Plot wrist with special color and size
        ax.scatter(coords[0, 0], coords[0, 1], 
                  color=finger_colors['wrist'], s=100, alpha=0.9, 
                  edgecolors='black', linewidth=2, zorder=10, label='Wrist')
        
        # Plot each finger with different colors
        for finger_name, (start_joint, end_joint) in finger_ranges.items():
            finger_joints = list(range(start_joint, end_joint + 1))
            finger_color = finger_colors[finger_name]
            
            # Plot finger joints
            ax.scatter(coords[finger_joints, 0], coords[finger_joints, 1], 
                      c=finger_color, s=60, alpha=0.8, edgecolors='black', 
                      linewidth=1, zorder=5, label=finger_name.capitalize())
            
            # Plot finger connections
            for i in range(len(finger_joints) - 1):
                parent = finger_joints[i]
                child = finger_joints[i + 1]
                ax.plot([coords[parent, 0], coords[child, 0]],
                       [coords[parent, 1], coords[child, 1]], 
                       color=finger_color, linewidth=3, alpha=0.8, zorder=4)
        
        # Plot connections from wrist to finger bases
        wrist_connections = [(0, 1), (0, 4), (0, 7), (0, 10), (0, 13)]
        for parent, child in wrist_connections:
            if parent < coords.shape[0] and child < coords.shape[0]:
                ax.plot([coords[parent, 0], coords[child, 0]],
                       [coords[parent, 1], coords[child, 1]], 
                       color=finger_colors['wrist'], linewidth=4, alpha=0.9, zorder=3)
        
        if show_labels:
            # Add joint labels for key joints
            joint_labels = {
                0: 'Wrist',
                1: 'T1', 4: 'I1', 7: 'M1', 10: 'R1', 13: 'L1',  # Finger bases
                3: 'T3', 6: 'I3', 9: 'M3', 12: 'R3', 15: 'L3'   # Finger tips
            }
            
            for joint_idx, label in joint_labels.items():
                if joint_idx < coords.shape[0]:
                    ax.text(coords[joint_idx, 0], coords[joint_idx, 1], 
                           label, fontsize=10, fontweight='bold', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.legend(loc='upper right', fontsize=8)
        
        # Set uniform tick spacing and grid
        self.set_uniform_axes_2d(ax)
        
        return ax
    
    def visualize_single_frame(self, frame_idx: int, save_path: Optional[str] = None):
        """
        Visualize a single frame with both hands.
        
        Args:
            frame_idx (int): Frame index to visualize
            save_path (str, optional): Path to save the visualization
        """
        left_hand, right_hand = self.get_frame_data(frame_idx)
        
        fig = plt.figure(figsize=(15, 8))
        
        if self.view_mode == '3d':
            # Create subplots for left and right hands (3D)
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            
            # Visualize hands in 3D
            visualize_hand_3d(left_hand, f"Left Hand - Frame {frame_idx}", ax1, self.show_labels)
            visualize_hand_3d(right_hand, f"Right Hand - Frame {frame_idx}", ax2, self.show_labels)
            
            # Set consistent view angles and uniform scaling
            for ax in [ax1, ax2]:
                ax.view_init(elev=20, azim=45)
                if self.auto_scale:
                    # Calculate bounds for consistent scaling
                    all_coords = np.vstack([left_hand, right_hand])
                    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
                    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
                    z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()
                    
                    # Add some padding
                    padding = 0.1
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    z_range = z_max - z_min
                    
                    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
                    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
                    ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
                
                # Set uniform tick spacing and grid
                self.set_uniform_axes(ax)
        
        else:  # 2D top-down view
            # Create subplots for left and right hands (2D)
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            
            # Visualize hands in 2D top-down view
            self.visualize_hand_2d_topdown(left_hand, f"Left Hand - Frame {frame_idx}", ax1, self.show_labels)
            self.visualize_hand_2d_topdown(right_hand, f"Right Hand - Frame {frame_idx}", ax2, self.show_labels)
            
            # Set consistent scaling for both plots
            if self.auto_scale:
                all_coords = np.vstack([left_hand, right_hand])
                x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
                y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
                
                # Add some padding
                padding = 0.1
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                for ax in [ax1, ax2]:
                    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
                    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
        
        view_mode_text = "3D" if self.view_mode == '3d' else "2D Top-Down"
        plt.suptitle(f'Hand Gesture Visualization - Frame {frame_idx} of {self.num_frames-1} ({view_mode_text})', 
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
        fig = plt.figure(figsize=(16, 10))
        
        # Create main visualization area
        if self.view_mode == '3d':
            ax_left = fig.add_subplot(2, 3, 1, projection='3d')
            ax_right = fig.add_subplot(2, 3, 2, projection='3d')
        else:
            ax_left = fig.add_subplot(2, 3, 1)
            ax_right = fig.add_subplot(2, 3, 2)
        
        ax_stats = fig.add_subplot(2, 3, 3)
        
        # Create control area
        ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
        ax_play = fig.add_axes([0.1, 0.15, 0.1, 0.04])
        ax_speed = fig.add_axes([0.25, 0.15, 0.15, 0.04])
        ax_export = fig.add_axes([0.45, 0.15, 0.1, 0.04])
        ax_stats_btn = fig.add_axes([0.6, 0.15, 0.1, 0.04])
        
        # Initialize visualization
        left_hand, right_hand = self.get_frame_data(0)
        
        # Create initial plots based on view mode
        if self.view_mode == '3d':
            visualize_hand_3d(left_hand, "Left Hand", ax_left, self.show_labels)
            visualize_hand_3d(right_hand, "Right Hand", ax_right, self.show_labels)
            
            # Set consistent view angles and uniform scaling
            for ax in [ax_left, ax_right]:
                ax.view_init(elev=20, azim=45)
                self.set_uniform_axes(ax)
        else:
            self.visualize_hand_2d_topdown(left_hand, "Left Hand", ax_left, self.show_labels)
            self.visualize_hand_2d_topdown(right_hand, "Right Hand", ax_right, self.show_labels)
        
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
            left_hand, right_hand = self.get_frame_data(frame_idx)
            
            # Clear previous plots
            ax_left.clear()
            ax_right.clear()
            
            # Redraw hands based on view mode
            if self.view_mode == '3d':
                visualize_hand_3d(left_hand, f"Left Hand - Frame {frame_idx}", ax_left, self.show_labels)
                visualize_hand_3d(right_hand, f"Right Hand - Frame {frame_idx}", ax_right, self.show_labels)
                
                # Set consistent view angles and uniform scaling
                for ax in [ax_left, ax_right]:
                    ax.view_init(elev=20, azim=45)
                    if self.auto_scale:
                        all_coords = np.vstack([left_hand, right_hand])
                        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
                        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
                        z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()
                        
                        padding = 0.1
                        x_range = x_max - x_min
                        y_range = y_max - y_min
                        z_range = z_max - z_min
                        
                        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
                        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
                        ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
                    
                    # Set uniform tick spacing and grid
                    self.set_uniform_axes(ax)
            else:
                self.visualize_hand_2d_topdown(left_hand, f"Left Hand - Frame {frame_idx}", ax_left, self.show_labels)
                self.visualize_hand_2d_topdown(right_hand, f"Right Hand - Frame {frame_idx}", ax_right, self.show_labels)
                
                # Set consistent scaling for both plots
                if self.auto_scale:
                    all_coords = np.vstack([left_hand, right_hand])
                    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
                    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
                    
                    padding = 0.1
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    
                    for ax in [ax_left, ax_right]:
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
        
        plt.suptitle('Interactive Hand Gesture Visualizer', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def create_animation(self, output_path: str = "hand_gesture_animation.gif", 
                        fps: int = 10, dpi: int = 100):
        """
        Create an animated GIF of the hand gestures.
        
        Args:
            output_path (str): Path to save the animation
            fps (int): Frames per second for the animation
            dpi (int): DPI for the output animation
        """
        fig = plt.figure(figsize=(15, 8))
        
        # Create subplots based on view mode
        if self.view_mode == '3d':
            ax_left = fig.add_subplot(1, 2, 1, projection='3d')
            ax_right = fig.add_subplot(1, 2, 2, projection='3d')
        else:
            ax_left = fig.add_subplot(1, 2, 1)
            ax_right = fig.add_subplot(1, 2, 2)
        
        def animate(frame_idx):
            """Animation function for each frame."""
            left_hand, right_hand = self.get_frame_data(frame_idx)
            
            # Clear previous plots
            ax_left.clear()
            ax_right.clear()
            
            # Visualize hands based on view mode
            if self.view_mode == '3d':
                visualize_hand_3d(left_hand, f"Left Hand - Frame {frame_idx}", ax_left, False)
                visualize_hand_3d(right_hand, f"Right Hand - Frame {frame_idx}", ax_right, False)
                
                # Set consistent view angles and uniform scaling
                for ax in [ax_left, ax_right]:
                    ax.view_init(elev=20, azim=45)
                    if self.auto_scale:
                        all_coords = np.vstack([left_hand, right_hand])
                        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
                        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
                        z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()
                        
                        padding = 0.1
                        x_range = x_max - x_min
                        y_range = y_max - y_min
                        z_range = z_max - z_min
                        
                        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
                        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
                        ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
                    
                    # Set uniform tick spacing and grid
                    self.set_uniform_axes(ax)
            else:
                self.visualize_hand_2d_topdown(left_hand, f"Left Hand - Frame {frame_idx}", ax_left, False)
                self.visualize_hand_2d_topdown(right_hand, f"Right Hand - Frame {frame_idx}", ax_right, False)
                
                # Set consistent scaling for both plots
                if self.auto_scale:
                    all_coords = np.vstack([left_hand, right_hand])
                    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
                    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
                    
                    padding = 0.1
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    
                    for ax in [ax_left, ax_right]:
                        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
                        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
            
            view_mode_text = "3D" if self.view_mode == '3d' else "2D Top-Down"
            plt.suptitle(f'Hand Gesture Animation - Frame {frame_idx} of {self.num_frames-1} ({view_mode_text})', 
                        fontsize=14, fontweight='bold')
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=self.num_frames, 
                                     interval=1000//fps, repeat=True)
        
        # Save animation
        print(f"Creating animation with {self.num_frames} frames at {fps} FPS...")
        anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
        print(f"Animation saved to {output_path}")
        
        plt.show()

def main():
    """Main function to run the hand gesture visualizer."""
    parser = argparse.ArgumentParser(description='Visualize hand gestures from JSON data')
    parser.add_argument('json_file', help='Path to the JSON file containing hand motion data')
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
    visualizer = HandGestureVisualizer(args.json_file, args.view)
    
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