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
- Piano keyboard visualization with key press detection
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
        
        # Validate joint structure - ensure we have 16 joints per hand
        if self.num_frames > 0:
            sample_frame = self.frames[0]
            left_joints = sample_frame.get('left_hand_joints', [])
            right_joints = sample_frame.get('right_hand_joints', [])
            
            if len(left_joints) != 16 or len(right_joints) != 16:
                print(f"Warning: Expected 16 joints per hand, but got {len(left_joints)} left and {len(right_joints)} right joints")
                print("Using 16-joint connection structure for visualization")
        
        # Extract metadata
        self.metadata = self.data.get('metadata', {})
        self.mano_joint_names = self.metadata.get('mano_joint_names', [])
        
        # Use correct 16-joint connections for the actual data structure
        # The data has 16 joints per hand, not 21
        self.mano_joint_connections = [
            (0, 1), (1, 2), (2, 3),      # Thumb
            (0, 4), (4, 5), (5, 6),      # Index
            (0, 7), (7, 8), (8, 9),      # Middle
            (0, 10), (10, 11), (11, 12), # Ring
            (0, 13), (13, 14), (14, 15)  # Little
        ]
        
        self.finger_groups = self.metadata.get('finger_groups', {})
        
        # Extract keyboard information
        self.keyboard_info = self.data.get('keyboard_info', {})
        self.keyboard_width = self.keyboard_info.get('total_width', 1.2)
        self.key_count = self.keyboard_info.get('key_count', 88)
        self.white_keys = self.keyboard_info.get('white_keys', 52)
        self.white_key_width = self.keyboard_info.get('white_key_width', 0.023)
        
        # Animation state
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0  # frames per second
        
        # Visualization settings
        self.show_labels = True
        self.show_connections = True
        self.auto_scale = True
        self.show_keyboard = True  # New setting for keyboard visualization
        
        print(f"Loaded {self.num_frames} frames from {json_file_path}")
        print(f"Data format: {self.metadata.get('data_format', 'unknown')}")
        print(f"Joints per hand: {self.metadata.get('joints_per_hand', 'unknown')}")
        print(f"View mode: {self.view_mode.upper()}")
        print(f"Keyboard info: {self.key_count} keys, {self.white_keys} white keys, width: {self.keyboard_width}m")
    
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
            - left_hand_joints: Left hand joint coordinates (16, 3)
            - right_hand_joints: Right hand joint coordinates (16, 3)
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
        
        # Get FPS from metadata if available, otherwise use default
        fps = self.data.get('metadata', {}).get('fps', 30.0)
        
        stats = {
            'total_frames': self.num_frames,
            'duration_seconds': self.num_frames / fps,  # Using detected FPS
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
            joints: Joint coordinates (16, 3) - 16-joint MANO structure
            connections: List of (parent, child) joint connections for 16-joint model
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
        
        # Highlight key joints (wrist and fingertips) - 16-joint structure
        key_joints = [0, 3, 6, 9, 12, 15]  # Wrist and fingertips for 16-joint model
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
            joints: Joint coordinates (16, 3) - 16-joint MANO structure
            connections: List of (parent, child) joint connections for 16-joint model
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
        
        # Highlight key joints - 16-joint structure
        key_joints = [0, 3, 6, 9, 12, 15]  # Wrist and fingertips for 16-joint model
        for joint_idx in key_joints:
            if joint_idx < len(joints):
                ax.scatter(joints[joint_idx, 0], joints[joint_idx, 1], 
                          c='black', s=80, alpha=0.8, marker='o', edgecolors=color)
        
        if show_labels and self.mano_joint_names:
            # Add labels for key joints - 16-joint structure
            key_labels = {0: 'Wrist', 3: 'Thumb', 6: 'Index', 9: 'Middle', 12: 'Ring', 15: 'Little'}
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
        
        pressed_keys = []
        
        fig = plt.figure(figsize=(12, 10))
        
        if self.view_mode == '3d':
            # Create single 3D plot for both hands
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot both hands on the same plot
            self.plot_hand_3d(left_hand, self.mano_joint_connections, 
                             f"Left Hand - Frame {frame_idx}", 'blue', ax, alpha=0.8)
            self.plot_hand_3d(right_hand, self.mano_joint_connections, 
                             f"Right Hand - Frame {frame_idx}", 'red', ax, alpha=0.8, add_to_existing=True)
            
            # Add piano keyboard - position it to match hand coordinates
            keyboard_z = 18.8  # Position keyboard to match hand Z coordinates (~18.8)
            self.plot_piano_keyboard(ax, pressed_keys, keyboard_y=-0.1, keyboard_z=keyboard_z)
            
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
            
            # Add piano keyboard - position it below the hands
            self.plot_piano_keyboard(ax, pressed_keys, keyboard_y=-0.1)
            
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
    
    def create_piano_keyboard(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Create piano keyboard layout with white and black keys.
        Middle C (C4) is centered at x = 0.
        
        Returns:
            Tuple of (white_keys, black_keys) where each key is a dict with position and note info
        """
        white_keys = []
        black_keys = []
        
        # Standard piano layout: C, D, E, F, G, A, B pattern
        white_key_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        black_key_notes = ['C#', 'D#', '', 'F#', 'G#', 'A#', '']
        
        # Calculate the offset to center middle C (C4) at x = 0
        # According to TargetProcessor: MIDI 60 (C4) = piano_note 39 (60-21=39)
        # In piano layout: A0(21)=0, A#0(22)=black, B0(23)=1, C1(24)=2...
        # C4(60) corresponds to white key index 39 in the 88-key system
        # But we only show 52 white keys, so we need to map accordingly
        # White keys: A0,B0,C1,D1,E1,F1,G1,A1,B1,C2... (pattern: A,B,C,D,E,F,G)
        # C4 is the 4th C, so it's at white key index: 2 + 7*3 = 23 (0-indexed)
        middle_c_white_index = 23  # Correct white key index for C4 in 52-key layout
        middle_c_offset = middle_c_white_index * self.white_key_width
        
        # Calculate key positions starting from the leftmost key
        current_x = -middle_c_offset  # Start so that C4 will be at x = 0
        octave = 0
        
        for i in range(self.white_keys):
            # White key
            note = white_key_notes[i % 7]
            full_note = f"{note}{octave + (i // 7)}"
            midi_note = 21 + i  # A0 = 21, C1 = 24, etc.
            
            white_keys.append({
                'x': current_x,
                'width': self.white_key_width,
                'note': full_note,
                'midi_note': midi_note,
                'octave': octave + (i // 7)
            })
            
            # Add black key if applicable (except between E-F and B-C)
            if i % 7 in [0, 1, 3, 4, 5]:  # C, D, F, G, A
                black_note = black_key_notes[i % 7]
                if black_note:  # Skip empty slots
                    # Calculate black key MIDI note more accurately
                    black_midi_note = midi_note + 1  # Sharp of the current white key
                    black_keys.append({
                        'x': current_x + self.white_key_width * 0.6,  # Position black key
                        'width': self.white_key_width * 0.6,
                        'note': f"{black_note}{octave + (i // 7)}",
                        'midi_note': black_midi_note,
                        'octave': octave + (i // 7)
                    })
            
            current_x += self.white_key_width
            
            # Increment octave every 7 white keys
            if (i + 1) % 7 == 0:
                octave += 1
        
        return white_keys, black_keys
    

    
    def plot_piano_keyboard(self, ax, pressed_keys: List[Dict] = None, 
                          keyboard_y: float = 0.05, keyboard_z: float = 18.8):
        """
        Plot piano keyboard in 3D space.
        
        Args:
            ax: Matplotlib axis (3D or 2D)
            pressed_keys: List of currently pressed keys
            keyboard_y: Y position of keyboard
            keyboard_z: Z position of keyboard
        """
        if not self.show_keyboard:
            return
            
        white_keys, black_keys = self.create_piano_keyboard()
        
        # Create sets of pressed key notes for quick lookup
        pressed_notes = set()
        if pressed_keys:
            pressed_notes = {key['key']['note'] for key in pressed_keys}
        
        # Plot white keys
        for key in white_keys:
            x = key['x']
            width = key['width']
            note = key['note']
            
            # Determine color based on whether key is pressed
            if note in pressed_notes:
                color = 'red'  # Pressed keys are red
                alpha = 0.8
            else:
                color = 'white'
                alpha = 0.6
            
            if self.view_mode == '3d':
                # 3D keyboard visualization
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                # Create 3D rectangle for key
                vertices = np.array([
                    [x, keyboard_y, keyboard_z],
                    [x + width, keyboard_y, keyboard_z],
                    [x + width, keyboard_y + 0.15, keyboard_z],  # Key length
                    [x, keyboard_y + 0.15, keyboard_z]
                ])
                
                # Create faces for 3D key
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # Top face
                    [vertices[0], vertices[1], vertices[1] + [0, 0, 0.02], vertices[0] + [0, 0, 0.02]],  # Front face
                    [vertices[1], vertices[2], vertices[2] + [0, 0, 0.02], vertices[1] + [0, 0, 0.02]],  # Right face
                    [vertices[2], vertices[3], vertices[3] + [0, 0, 0.02], vertices[2] + [0, 0, 0.02]],  # Back face
                    [vertices[3], vertices[0], vertices[0] + [0, 0, 0.02], vertices[3] + [0, 0, 0.02]]   # Left face
                ]
                
                poly3d = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_collection3d(poly3d)
                
                # Add note label
                ax.text(x + width/2, keyboard_y + 0.075, keyboard_z + 0.03, 
                       note, fontsize=8, ha='center', va='center')
            else:
                # 2D keyboard visualization
                rect = plt.Rectangle((x, keyboard_y), width, 0.15, 
                                   facecolor=color, edgecolor='black', 
                                   linewidth=1, alpha=alpha)
                ax.add_patch(rect)
                
                # Add note label
                ax.text(x + width/2, keyboard_y + 0.075, note, 
                       fontsize=8, ha='center', va='center')
        
        # Plot black keys
        for key in black_keys:
            x = key['x']
            width = key['width']
            note = key['note']
            
            # Determine color based on whether key is pressed
            if note in pressed_notes:
                color = 'darkred'  # Pressed black keys are dark red
                alpha = 0.9
            else:
                color = 'black'
                alpha = 0.8
            
            if self.view_mode == '3d':
                # 3D black key visualization
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                # Create 3D rectangle for black key
                vertices = np.array([
                    [x, keyboard_y, keyboard_z],
                    [x + width, keyboard_y, keyboard_z],
                    [x + width, keyboard_y + 0.095, keyboard_z],  # Black key length
                    [x, keyboard_y + 0.095, keyboard_z]
                ])
                
                # Create faces for 3D black key
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # Top face
                    [vertices[0], vertices[1], vertices[1] + [0, 0, 0.02], vertices[0] + [0, 0, 0.02]],  # Front face
                    [vertices[1], vertices[2], vertices[2] + [0, 0, 0.02], vertices[1] + [0, 0, 0.02]],  # Right face
                    [vertices[2], vertices[3], vertices[3] + [0, 0, 0.02], vertices[2] + [0, 0, 0.02]],  # Back face
                    [vertices[3], vertices[0], vertices[0] + [0, 0, 0.02], vertices[3] + [0, 0, 0.02]]   # Left face
                ]
                
                poly3d = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='gray', linewidth=0.5)
                ax.add_collection3d(poly3d)
                
                # Add note label
                ax.text(x + width/2, keyboard_y + 0.0475, keyboard_z + 0.03, 
                       note, fontsize=6, ha='center', va='center', color='white')
            else:
                # 2D black key visualization
                rect = plt.Rectangle((x, keyboard_y), width, 0.095, 
                                   facecolor=color, edgecolor='gray', 
                                   linewidth=1, alpha=alpha)
                ax.add_patch(rect)
                
                # Add note label
                ax.text(x + width/2, keyboard_y + 0.0475, note, 
                       fontsize=6, ha='center', va='center', color='white')

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
        ax_keyboard_btn = fig.add_axes([0.75, 0.15, 0.1, 0.04])
        
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
        keyboard_button = Button(ax_keyboard_btn, 'Keyboard: ON')
        
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
            
            pressed_keys = []
            
            # Clear previous plots
            ax.clear()
            
            # Redraw both hands based on view mode
            if self.view_mode == '3d':
                self.plot_hand_3d(left_hand, self.mano_joint_connections, 
                                f"Left Hand - Frame {frame_idx}", 'blue', ax, alpha=0.8)
                self.plot_hand_3d(right_hand, self.mano_joint_connections, 
                                f"Right Hand - Frame {frame_idx}", 'red', ax, alpha=0.8, add_to_existing=True)
                
                # Add piano keyboard
                self.plot_piano_keyboard(ax, pressed_keys, keyboard_y=-0.1, keyboard_z=18.8)
                
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
                
                # Add piano keyboard
                self.plot_piano_keyboard(ax, pressed_keys, keyboard_y=-0.1)
                
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
        
        def toggle_keyboard(event):
            """Toggle keyboard visualization on/off."""
            nonlocal self
            self.show_keyboard = not self.show_keyboard
            keyboard_button.label.set_text(f'Keyboard: {"ON" if self.show_keyboard else "OFF"}')
            # Update the current frame to show/hide keyboard
            update_frame(slider.val)
        
        # Connect events
        slider.on_changed(update_frame)
        play_button.on_clicked(play_animation)
        speed_button.on_clicked(change_speed)
        export_button.on_clicked(export_frame)
        stats_button.on_clicked(show_stats)
        keyboard_button.on_clicked(toggle_keyboard)
        
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
            
            pressed_keys = []
            
            # Clear previous plots
            ax.clear()
            
            # Visualize both hands based on view mode
            if self.view_mode == '3d':
                self.plot_hand_3d(left_hand, self.mano_joint_connections, 
                                f"Left Hand - Frame {frame_idx}", 'blue', ax, alpha=0.8)
                self.plot_hand_3d(right_hand, self.mano_joint_connections, 
                                f"Right Hand - Frame {frame_idx}", 'red', ax, alpha=0.8, add_to_existing=True)
                
                # Add piano keyboard
                self.plot_piano_keyboard(ax, pressed_keys, keyboard_y=-0.1, keyboard_z=18.8)
                
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
                
                # Add piano keyboard
                self.plot_piano_keyboard(ax, pressed_keys, keyboard_y=-0.1)
                
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
    parser.add_argument('--no-keyboard', action='store_true', 
                       help='Disable keyboard visualization')
    
    args = parser.parse_args()
    
    # Create visualizer with specified view mode
    visualizer = StandaloneHandGestureVisualizer(args.json_file, args.view)
    
    # Set keyboard visibility
    if args.no_keyboard:
        visualizer.show_keyboard = False
    
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