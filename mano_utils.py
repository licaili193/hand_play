"""
MANO (Model of Articulated Objects) utility functions for hand joint analysis and visualization.

This module provides functions to:
- Extract anatomically precise joint connections from MANO model
- Map between different joint representations (16-joint to 21-joint)
- Provide finger-specific colors and ranges for visualization
- Analyze and print MANO joint structure information
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import torch # Added for MANO forward kinematics

# Import function from midi_to_frames.py
try:
    from midi_to_frames import process_model_output_with_proper_coordinates
except ImportError:
    # Fallback if the function is not available
    def process_model_output_with_proper_coordinates(pose_hat, guide, device='cpu'):
        """
        Fallback implementation if the main function is not available.
        """
        print("Warning: Using fallback process_model_output_with_proper_coordinates")
        
        # Simple processing without coordinate transformations
        prediction = pose_hat[0].detach().cpu().numpy()
        scaled_guide = guide[0].detach().cpu().numpy()
        
        # Split data
        right_hand_angles = prediction[:, :48]
        left_hand_angles = prediction[:, 48:]
        right_hand_pos = scaled_guide[:, :3]
        left_hand_pos = scaled_guide[:, 3:]
        
        return {
            'right_hand_angles': right_hand_angles,
            'left_hand_angles': left_hand_angles,
            'right_hand_position': right_hand_pos,
            'left_hand_position': left_hand_pos,
            'num_frames': prediction.shape[0],
            'coordinate_info': {},
            'spatial_info': {},
            'keyboard_info': {}
        }

def find_mano_model_path():
    """
    Find the path to the MANO model directory.
    
    Returns:
        str: Path to the MANO model directory, or None if not found
    """
    # Check common locations for the MANO model
    possible_paths = [
        'PianoMotion10M/mano',
        './PianoMotion10M/mano',
        '../PianoMotion10M/mano',
        'mano',
        './mano'
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if MANO_RIGHT.pkl exists in this directory
            mano_file = os.path.join(path, 'MANO_RIGHT.pkl')
            if os.path.exists(mano_file):
                print(f"Found MANO model at: {os.path.abspath(path)}")
                return path
    
    return None

def get_mano_joint_connections():
    """
    Extract anatomically precise joint connections from MANO model.
    Returns the proper kinematic tree for hand visualization.
    
    Returns:
        list: List of (parent, child) joint connections
    """
    try:
        # Import MANO model to get proper joint hierarchy
        import sys
        import os
        sys.path.append('PianoMotion10M')
        from models.mano import build_mano
        
        # Find the MANO model path
        mano_path = find_mano_model_path()
        if not mano_path:
            raise FileNotFoundError("MANO model directory not found. Please ensure MANO_RIGHT.pkl is in PianoMotion10M/mano/")
        
        # Change to the PianoMotion10M directory to ensure correct path resolution
        original_cwd = os.getcwd()
        piano_motion_dir = os.path.join(original_cwd, 'PianoMotion10M')
        
        if not os.path.exists(piano_motion_dir):
            raise FileNotFoundError(f"PianoMotion10M directory not found at {piano_motion_dir}")
        
        # Change to PianoMotion10M directory temporarily
        os.chdir(piano_motion_dir)
        
        try:
            # Load MANO model (now it will look for ./mano relative to PianoMotion10M)
            mano_layer = build_mano()
            mano_model = mano_layer['right']
            
            # Explore the MANO model structure to find the kinematic tree
            print("Exploring MANO model structure...")
            
            # Check available attributes
            available_attrs = [attr for attr in dir(mano_model) if not attr.startswith('_')]
            print(f"Available attributes: {available_attrs}")
            
            # Try different possible attribute names for kinematic tree
            kinematic_tree = None
            possible_names = ['kinematic_tree', 'parents', 'joint_parents', 'joint_tree', 'parent_idx']
            
            for attr_name in possible_names:
                if hasattr(mano_model, attr_name):
                    kinematic_tree = getattr(mano_model, attr_name)
                    print(f"Found kinematic tree at attribute: {attr_name}")
                    # Convert tensor to list if needed
                    if hasattr(kinematic_tree, 'tolist'):
                        kinematic_tree = kinematic_tree.tolist()
                    elif hasattr(kinematic_tree, 'cpu'):
                        kinematic_tree = kinematic_tree.cpu().numpy().tolist()
                    break
            
            # If we still don't have a kinematic tree, try to get it from the model's joints
            if kinematic_tree is None:
                print("Kinematic tree not found in attributes, using standard MANO joint connections...")
                # Use the standard MANO joint connections (21 joints)
                kinematic_tree = [
                    -1,  # 0: Wrist (root)
                    0,   # 1: Thumb_CMC
                    1,   # 2: Thumb_MCP
                    2,   # 3: Thumb_IP
                    3,   # 4: Thumb_Tip
                    0,   # 5: Index_MCP
                    5,   # 6: Index_PIP
                    6,   # 7: Index_DIP
                    7,   # 8: Index_Tip
                    0,   # 9: Middle_MCP
                    9,   # 10: Middle_PIP
                    10,  # 11: Middle_DIP
                    11,  # 12: Middle_Tip
                    0,   # 13: Ring_MCP
                    13,  # 14: Ring_PIP
                    14,  # 15: Ring_DIP
                    15,  # 16: Ring_Tip
                    0,   # 17: Little_MCP
                    17,  # 18: Little_PIP
                    18,  # 19: Little_DIP
                    19   # 20: Little_Tip
                ]
            
            # Convert kinematic tree to list of connections
            connections = []
            for child_idx, parent_idx in enumerate(kinematic_tree):
                if parent_idx != -1:  # -1 indicates root (wrist)
                    connections.append((parent_idx, child_idx))
            
            print(f"Extracted {len(connections)} connections from MANO model")
            
            # Print detailed MANO joint information
            print_mano_joint_info(mano_model, kinematic_tree)
            
            return connections
            
        finally:
            # Always restore the original working directory
            os.chdir(original_cwd)
        
    except Exception as e:
        print(f"Could not load MANO model for precise connections: {e}")
        print("Falling back to manual connections...")
        print("To use precise MANO connections, ensure:")
        print("1. MANO_RIGHT.pkl is in PianoMotion10M/mano/ directory")
        print("2. You're running the script from the project root directory")
        
        # Fallback to manual connections (simplified 16-joint version)
        # This matches the 16 joints per hand that your model outputs
        manual_connections = [
            # Wrist to finger bases
            (0, 1), (0, 4), (0, 7), (0, 10), (0, 13),
            
            # Thumb (3 joints)
            (1, 2), (2, 3),
            
            # Index finger (3 joints)
            (4, 5), (5, 6),
            
            # Middle finger (3 joints)
            (7, 8), (8, 9),
            
            # Ring finger (3 joints)
            (10, 11), (11, 12),
            
            # Little finger (3 joints)
            (13, 14), (14, 15)
        ]
        return manual_connections

def print_mano_joint_info(mano_model, kinematic_tree=None):
    """
    Print detailed information about MANO joint structure.
    This helps understand the mapping between model output and MANO joints.
    
    Args:
        mano_model: Loaded MANO model instance
        kinematic_tree: Optional kinematic tree (if not found in model)
    """
    print("\n=== MANO Joint Structure Analysis ===")
    
    # Try to get number of joints
    num_joints = None
    if hasattr(mano_model, 'num_joints'):
        num_joints = mano_model.num_joints
    elif hasattr(mano_model, 'NUM_JOINTS'):
        num_joints = mano_model.NUM_JOINTS
    elif hasattr(mano_model, 'NUM_HAND_JOINTS'):
        num_joints = mano_model.NUM_HAND_JOINTS
    elif hasattr(mano_model, 'num_vertices'):
        # Sometimes joints are derived from vertices
        num_joints = 21  # Standard MANO has 21 joints
    elif kinematic_tree:
        # Use the length of kinematic tree as number of joints
        num_joints = len(kinematic_tree)
    else:
        num_joints = 21  # Default to standard MANO
    
    print(f"Total MANO joints: {num_joints}")
    
    if kinematic_tree:
        print(f"Kinematic tree length: {len(kinematic_tree)}")
    
    # MANO joint names (standard MANO joint ordering)
    mano_joint_names = [
        'Wrist',           # 0
        'Thumb_CMC',       # 1
        'Thumb_MCP',       # 2
        'Thumb_IP',        # 3
        'Thumb_Tip',       # 4
        'Index_MCP',       # 5
        'Index_PIP',       # 6
        'Index_DIP',       # 7
        'Index_Tip',       # 8
        'Middle_MCP',      # 9
        'Middle_PIP',      # 10
        'Middle_DIP',      # 11
        'Middle_Tip',      # 12
        'Ring_MCP',        # 13
        'Ring_PIP',        # 14
        'Ring_DIP',        # 15
        'Ring_Tip',        # 16
        'Little_MCP',      # 17
        'Little_PIP',      # 18
        'Little_DIP',      # 19
        'Little_Tip'       # 20
    ]
    
    if kinematic_tree:
        print("\nMANO Joint Names:")
        for i, name in enumerate(mano_joint_names):
            if i < len(kinematic_tree):
                parent = kinematic_tree[i]
                parent_name = mano_joint_names[parent] if parent != -1 and parent < len(mano_joint_names) else "Root"
                print(f"  {i:2d}: {name:12s} -> Parent: {parent_name}")
    
    print(f"\nNote: Your model outputs 16 joints per hand, while MANO has {num_joints} joints.")
    print("The model likely uses a subset or simplified version of the MANO joint structure.")
    print("=" * 50)

def get_16_to_21_joint_mapping():
    """
    Create a mapping from 16-joint model output to 21-joint MANO structure.
    This is a reasonable approximation based on common hand modeling practices.
    
    Returns:
        dict: Mapping from 16-joint indices to 21-joint MANO indices
    """
    # Mapping from 16-joint indices to 21-joint MANO indices
    # This assumes the model outputs key joints and skips some intermediate ones
    mapping_16_to_21 = {
        # Wrist
        0: 0,   # Wrist
        
        # Thumb (3 joints in model -> 4 joints in MANO)
        1: 1,   # Thumb base (CMC)
        2: 2,   # Thumb middle (MCP) 
        3: 4,   # Thumb tip (skip IP, go to tip)
        
        # Index finger (3 joints in model -> 4 joints in MANO)
        4: 5,   # Index base (MCP)
        5: 6,   # Index middle (PIP)
        6: 8,   # Index tip (skip DIP, go to tip)
        
        # Middle finger (3 joints in model -> 4 joints in MANO)
        7: 9,   # Middle base (MCP)
        8: 10,  # Middle middle (PIP)
        9: 12,  # Middle tip (skip DIP, go to tip)
        
        # Ring finger (3 joints in model -> 4 joints in MANO)
        10: 13, # Ring base (MCP)
        11: 14, # Ring middle (PIP)
        12: 16, # Ring tip (skip DIP, go to tip)
        
        # Little finger (3 joints in model -> 4 joints in MANO)
        13: 17, # Little base (MCP)
        14: 18, # Little middle (PIP)
        15: 20  # Little tip (skip DIP, go to tip)
    }
    
    return mapping_16_to_21

def get_finger_colors():
    """
    Define colors for different fingers in visualization.
    
    Returns:
        dict: Color mapping for each finger
    """
    return {
        'wrist': '#2C3E50',      # Dark blue-gray
        'thumb': '#FF6B6B',      # Red
        'index': '#4ECDC4',      # Teal
        'middle': '#45B7D1',     # Blue
        'ring': '#96CEB4',       # Green
        'little': '#FFEAA7'      # Yellow
    }

def get_finger_ranges():
    """
    Define joint ranges for each finger in the 16-joint model.
    
    Returns:
        dict: Joint ranges for each finger (start, end)
    """
    return {
        'thumb': (1, 3),    # Joints 1-3
        'index': (4, 6),    # Joints 4-6
        'middle': (7, 9),   # Joints 7-9
        'ring': (10, 12),   # Joints 10-12
        'little': (13, 15)  # Joints 13-15
    }

def get_mano_joint_names():
    """
    Get the standard MANO joint names.
    
    Returns:
        list: List of MANO joint names
    """
    return [
        'Wrist', 'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip',
        'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
        'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
        'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
        'Little_MCP', 'Little_PIP', 'Little_DIP', 'Little_Tip'
    ]

def save_joint_mapping_info(total_joints, joints_per_hand, output_path="joint_mapping_info.json"):
    """
    Save joint mapping information to a JSON file for future reference.
    
    Args:
        total_joints (int): Total number of joints in model output
        joints_per_hand (int): Number of joints per hand
        output_path (str): Path to save the mapping information
    """
    joint_mapping_info = {
        "model_output_structure": {
            "total_joints": total_joints,
            "joints_per_hand": joints_per_hand,
            "description": f"{joints_per_hand} joints per hand, simplified version of MANO"
        },
        "16_to_21_mapping": get_16_to_21_joint_mapping(),
        "finger_ranges": get_finger_ranges(),
        "mano_joint_names": get_mano_joint_names()
    }
    
    with open(output_path, "w") as f:
        json.dump(joint_mapping_info, f, indent=2)
    print(f"Saved joint mapping information to {output_path}")

def set_uniform_axes_3d(ax):
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

def visualize_hand_3d(coords, title="Hand", ax=None, show_labels=True):
    """
    Visualize a single hand in 3D with proper finger coloring and connections.
    
    Args:
        coords (np.ndarray): Joint coordinates of shape (num_joints, 3)
        title (str): Title for the plot
        ax (matplotlib.axes.Axes3D, optional): Existing 3D axis to plot on
        show_labels (bool): Whether to show joint labels
        
    Returns:
        matplotlib.axes.Axes3D: The axis object
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    
    finger_colors = get_finger_colors()
    finger_ranges = get_finger_ranges()
    
    # Plot wrist with special color and size
    ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], 
              color=finger_colors['wrist'], s=100, alpha=0.9, 
              edgecolors='black', linewidth=2, zorder=10, label='Wrist')
    
    # Plot each finger with different colors
    for finger_name, (start_joint, end_joint) in finger_ranges.items():
        finger_joints = list(range(start_joint, end_joint + 1))
        finger_color = finger_colors[finger_name]
        
        # Plot finger joints
        ax.scatter(coords[finger_joints, 0], coords[finger_joints, 1], coords[finger_joints, 2], 
                  c=finger_color, s=60, alpha=0.8, edgecolors='black', 
                  linewidth=1, zorder=5, label=finger_name.capitalize())
        
        # Plot finger connections
        for i in range(len(finger_joints) - 1):
            parent = finger_joints[i]
            child = finger_joints[i + 1]
            ax.plot([coords[parent, 0], coords[child, 0]],
                   [coords[parent, 1], coords[child, 1]],
                   [coords[parent, 2], coords[child, 2]], 
                   color=finger_color, linewidth=3, alpha=0.8, zorder=4)
    
    # Plot connections from wrist to finger bases
    wrist_connections = [(0, 1), (0, 4), (0, 7), (0, 10), (0, 13)]
    for parent, child in wrist_connections:
        if parent < coords.shape[0] and child < coords.shape[0]:
            ax.plot([coords[parent, 0], coords[child, 0]],
                   [coords[parent, 1], coords[child, 1]],
                   [coords[parent, 2], coords[child, 2]], 
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
                ax.text(coords[joint_idx, 0], coords[joint_idx, 1], coords[joint_idx, 2], 
                       label, fontsize=10, fontweight='bold', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(loc='upper right', fontsize=8)
    
    # Set uniform tick spacing and grid
    set_uniform_axes_3d(ax)
    
    return ax

def visualize_both_hands_3d(left_hand_joints, right_hand_joints, frame_idx=0, total_frames=1):
    """
    Visualize both hands in 3D with proper finger coloring and connections.
    
    Args:
        left_hand_joints (list): Left hand joint coordinates
        right_hand_joints (list): Right hand joint coordinates
        frame_idx (int): Current frame index
        total_frames (int): Total number of frames
    """
    fig = plt.figure(figsize=(12, 6))
    
    # Plot left hand
    left_coords = np.array(left_hand_joints)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    visualize_hand_3d(left_coords, "Left Hand", ax1)
    
    # Plot right hand
    right_coords = np.array(right_hand_joints)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    visualize_hand_3d(right_coords, "Right Hand", ax2)
    
    # Set uniform axes for both plots
    set_uniform_axes_3d(ax1)
    set_uniform_axes_3d(ax2)
    
    plt.suptitle(f'Hand Motion Visualization - Frame {frame_idx} of {total_frames-1}\nUsing Anatomically Precise MANO Joint Connections', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.show() 

def visualize_with_correct_coordinates(processed_data, frame_idx=0):
    """
    Visualize hand motion with correct coordinate system handling.
    """
    
    # Get coordinate system information
    coord_info = processed_data.get('coordinate_info', {})
    
    # Extract data for visualization
    right_pos = processed_data['right_hand_position'][frame_idx]
    left_pos = processed_data['left_hand_position'][frame_idx]
    right_angles = processed_data['right_hand_angles'][frame_idx]
    left_angles = processed_data['left_hand_angles'][frame_idx]
    
    print(f"Visualization Frame {frame_idx}:")
    print(f"  Right hand position: [{right_pos[0]:.3f}, {right_pos[1]:.3f}, {right_pos[2]:.3f}]")
    print(f"  Left hand position: [{left_pos[0]:.3f}, {left_pos[1]:.3f}, {left_pos[2]:.3f}]")
    print(f"  Right hand angles range: [{np.min(right_angles):.3f}, {np.max(right_angles):.3f}] rad")
    print(f"  Left hand angles range: [{np.min(left_angles):.3f}, {np.max(left_angles):.3f}] rad")
    
    # Create 3D visualization with proper coordinate system
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Hand positions in world space
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(right_pos[0], right_pos[1], right_pos[2], 
               c='red', s=100, label='Right Hand', alpha=0.8)
    ax1.scatter(left_pos[0], left_pos[1], left_pos[2], 
               c='blue', s=100, label='Left Hand', alpha=0.8)
    
    ax1.set_xlabel('X (Lateral)')
    ax1.set_ylabel('Y (Vertical)')
    ax1.set_zlabel('Z (Depth)')
    ax1.set_title('Hand Positions\n(Scaled World Coordinates)')
    ax1.legend()
    
    # Plot 2: Angle distributions
    ax2 = fig.add_subplot(132)
    ax2.hist(right_angles, bins=20, alpha=0.7, label='Right Hand', color='red')
    ax2.hist(left_angles, bins=20, alpha=0.7, label='Left Hand', color='blue')
    ax2.set_xlabel('Angle (radians)')
    ax2.set_ylabel('Count')
    ax2.set_title('Hand Joint Angle Distribution')
    ax2.legend()
    
    # Plot 3: Coordinate system reference
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Draw coordinate system axes
    origin = np.array([0, 0, 0])
    axes_length = 10
    
    # X-axis (red)
    ax3.quiver(origin[0], origin[1], origin[2], axes_length, 0, 0, 
              color='red', arrow_length_ratio=0.1, label='X (Lateral)')
    # Y-axis (green)
    ax3.quiver(origin[0], origin[1], origin[2], 0, axes_length, 0, 
              color='green', arrow_length_ratio=0.1, label='Y (Vertical)')
    # Z-axis (blue)
    ax3.quiver(origin[0], origin[1], origin[2], 0, 0, axes_length, 
              color='blue', arrow_length_ratio=0.1, label='Z (Depth)')
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Coordinate System\nReference')
    ax3.legend()
    
    plt.tight_layout()
    plt.suptitle(f'PianoMotion10M Coordinate System Analysis - Frame {frame_idx}', y=1.02)
    plt.show()

def visualize_spatial_analysis(processed_data):
    """
    Visualize spatial analysis of hand movements with piano keyboard context.
    """
    
    spatial_info = processed_data.get('spatial_info', {})
    keyboard_info = processed_data.get('keyboard_info', {})
    
    if not spatial_info or not keyboard_info:
        print("Spatial analysis data not available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Hand movement ranges
    right_range = spatial_info.get('right_range', [0, 0, 0])
    left_range = spatial_info.get('left_range', [0, 0, 0])
    
    x_labels = ['X (Lateral)', 'Y (Vertical)', 'Z (Depth)']
    x_pos = np.arange(len(x_labels))
    width = 0.35
    
    axes[0, 0].bar(x_pos - width/2, right_range, width, label='Right Hand', color='red', alpha=0.7)
    axes[0, 0].bar(x_pos + width/2, left_range, width, label='Left Hand', color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('Coordinate Axis')
    axes[0, 0].set_ylabel('Movement Range (units)')
    axes[0, 0].set_title('Hand Movement Ranges')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(x_labels)
    axes[0, 0].legend()
    
    # Plot 2: Physical scale estimation
    right_physical = spatial_info.get('right_physical', [0, 0, 0])
    left_physical = spatial_info.get('left_physical', [0, 0, 0])
    
    axes[0, 1].bar(x_pos - width/2, right_physical, width, label='Right Hand', color='red', alpha=0.7)
    axes[0, 1].bar(x_pos + width/2, left_physical, width, label='Left Hand', color='blue', alpha=0.7)
    axes[0, 1].set_xlabel('Coordinate Axis')
    axes[0, 1].set_ylabel('Estimated Physical Range (meters)')
    axes[0, 1].set_title('Estimated Physical Scale')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(x_labels)
    axes[0, 1].legend()
    
    # Plot 3: Piano keyboard context
    keyboard_width = keyboard_info.get('total_width', 1.2)
    octave_width = keyboard_info.get('approximate_octave_width', 0.161)
    white_key_width = keyboard_info.get('white_key_width', 0.023)
    
    # Create a simple piano keyboard visualization
    keys = []
    colors = []
    for i in range(52):  # White keys
        keys.append(i * white_key_width)
        colors.append('white')
    
    axes[1, 0].barh(keys, [white_key_width] * 52, color=colors, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Position (meters)')
    axes[1, 0].set_ylabel('Key Index')
    axes[1, 0].set_title('Piano Keyboard Layout (White Keys)')
    axes[1, 0].set_xlim(0, keyboard_width)
    
    # Plot 4: Scale factors visualization
    scale_factors = spatial_info.get('scale_factors', [1.5, 1.5, 25])
    axes[1, 1].bar(x_labels, scale_factors, color=['red', 'green', 'blue'], alpha=0.7)
    axes[1, 1].set_ylabel('Scale Factor')
    axes[1, 1].set_title('Coordinate System Scale Factors')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.suptitle('PianoMotion10M Spatial Analysis', y=0.98)
    plt.show() 

def explain_pianomotion_mano_integration():
    """
    Explain how PianoMotion10M integrates with MANO model.
    """
    
    integration_info = {
        "model_architecture": {
            "position_predictor": "Outputs 6D (3D position for each hand)",
            "gesture_generator": "Outputs 96D (48 rotation parameters per hand)",
            "total_output": "6D position + 96D rotation = 102D per frame"
        },
        "mano_usage": {
            "parameterization": "MANO-style rotation parameters",
            "joint_count": "21 joints in full MANO, but parameterized differently",
            "output_format": "48 parameters per hand (not 48/3 = 16 joints)",
            "coordinate_system": "MANO coordinate frame with piano-specific adaptations"
        },
        "data_flow": {
            "step_1": "Audio → Position Predictor → 3D hand positions",
            "step_2": "Audio + Positions → Diffusion Model → Hand rotation parameters",
            "step_3": "Positions + Rotations → MANO Forward Kinematics → Joint positions",
            "step_4": "Joint positions → Visualization/Animation"
        },
        "official_rendering": {
            "method": "Uses render_result() from datasets/show.py",
            "input_format": "Concatenated [position, rotation] arrays",
            "right_hand": "np.concatenate([guide[:, :3], prediction[:, :48]], 1)",
            "left_hand": "np.concatenate([guide[:, 3:], prediction[:, 48:]], 1)"
        }
    }
    
    print("PianoMotion10M MANO Integration:")
    for category, details in integration_info.items():
        print(f"\n  {category.replace('_', ' ').title()}:")
        for key, value in details.items():
            print(f"    - {key.replace('_', ' ').title()}: {value}")
    
    return integration_info

def convert_mano_params_to_joints(hand_position, hand_rotations, hand='right'):
    """
    Convert MANO parameters (position + rotations) to joint positions.
    
    Args:
        hand_position: (3,) array - 3D position of hand
        hand_rotations: (48,) array - MANO rotation parameters
        hand: 'right' or 'left'
    
    Returns:
        joint_positions: (21, 3) array - 3D positions of MANO joints
    """
    
    try:
        # Try to use the actual MANO model if available
        import sys
        import os
        sys.path.append('PianoMotion10M')
        from models.mano import build_mano
        
        # Load MANO model
        mano_layer = build_mano()
        mano_model = mano_layer[hand]
        
        # Convert rotation parameters to MANO format
        # Note: This is a simplified conversion - actual implementation depends on
        # how PianoMotion10M parameterizes the rotations
        
        # Reshape rotation parameters (48,) → appropriate MANO format
        # This is dataset-specific and may require analysis of training code
        if len(hand_rotations) == 48:
            # Assume the 48 parameters map to MANO pose parameters
            # This mapping needs to be determined from the training code
            mano_pose = hand_rotations.reshape(-1)  # Keep as is for now
        else:
            raise ValueError(f"Unexpected rotation parameter count: {len(hand_rotations)}")
        
        # Set hand shape to mean (or zeros for simplicity)
        batch_size = 1
        hand_shape = torch.zeros(batch_size, 10)  # MANO shape parameters
        
        # Convert position to global translation
        global_orient = torch.zeros(batch_size, 3)  # Root rotation
        transl = torch.tensor(hand_position).unsqueeze(0).float()  # Translation
        
        # Convert pose parameters to tensor
        hand_pose = torch.tensor(mano_pose).unsqueeze(0).float()
        
        # Forward pass through MANO
        output = mano_model(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=hand_shape,
            transl=transl
        )
        
        # Extract joint positions
        joints = output.joints[0].detach().numpy()  # (21, 3)
        
        print(f"✓ MANO forward kinematics successful for {hand} hand")
        print(f"  - Input rotations: {hand_rotations.shape}")
        print(f"  - Input position: {hand_position}")
        print(f"  - Output joints: {joints.shape}")
        
        return joints
        
    except Exception as e:
        print(f"✗ MANO forward kinematics failed: {e}")
        print("Falling back to simplified joint approximation")
        
        # Fallback: Create approximate joint positions
        return approximate_joints_from_parameters(hand_position, hand_rotations, hand)

def approximate_joints_from_parameters(hand_position, hand_rotations, hand='right'):
    """
    Create approximate joint positions when full MANO is not available.
    """
    
    # Create a simplified hand model with 21 joints
    # This is an approximation - not as accurate as full MANO
    
    # Basic hand structure (relative to wrist)
    if hand == 'right':
        hand_sign = 1
    else:
        hand_sign = -1  # Mirror for left hand
    
    # Approximate joint offsets (in hand coordinate system)
    joint_offsets = np.array([
        [0, 0, 0],                           # 0: Wrist
        [hand_sign * 0.02, 0.01, 0.02],     # 1: Thumb CMC
        [hand_sign * 0.03, 0.02, 0.04],     # 2: Thumb MCP
        [hand_sign * 0.035, 0.03, 0.055],   # 3: Thumb IP
        [hand_sign * 0.04, 0.035, 0.07],    # 4: Thumb Tip
        [hand_sign * 0.02, 0.08, 0.01],     # 5: Index MCP
        [hand_sign * 0.02, 0.11, 0.015],    # 6: Index PIP
        [hand_sign * 0.02, 0.135, 0.02],    # 7: Index DIP
        [hand_sign * 0.02, 0.155, 0.025],   # 8: Index Tip
        [0, 0.09, 0.005],                   # 9: Middle MCP
        [0, 0.125, 0.01],                   # 10: Middle PIP
        [0, 0.15, 0.015],                   # 11: Middle DIP
        [0, 0.17, 0.02],                    # 12: Middle Tip
        [hand_sign * -0.02, 0.085, 0],      # 13: Ring MCP
        [hand_sign * -0.02, 0.115, 0.005],  # 14: Ring PIP
        [hand_sign * -0.02, 0.14, 0.01],    # 15: Ring DIP
        [hand_sign * -0.02, 0.16, 0.015],   # 16: Ring Tip
        [hand_sign * -0.04, 0.075, -0.005], # 17: Little MCP
        [hand_sign * -0.04, 0.1, 0],        # 18: Little PIP
        [hand_sign * -0.04, 0.12, 0.005],   # 19: Little DIP
        [hand_sign * -0.04, 0.135, 0.01]    # 20: Little Tip
    ])
    
    # Apply some rotation based on rotation parameters
    # This is a very simplified approach
    if len(hand_rotations) >= 6:  # At least some rotation info
        # Use first few rotation parameters for basic hand orientation
        rotation_factor = np.mean(hand_rotations[:6]) * 0.1  # Scale down
        
        # Simple rotation around Y-axis (finger curl)
        cos_r = np.cos(rotation_factor)
        sin_r = np.sin(rotation_factor)
        
        for i in range(len(joint_offsets)):
            x, y, z = joint_offsets[i]
            joint_offsets[i, 0] = x * cos_r - z * sin_r
            joint_offsets[i, 2] = x * sin_r + z * cos_r
    
    # Translate to world position
    joints = joint_offsets + hand_position
    
    print(f"✓ Created approximate joints for {hand} hand ({len(joints)} joints)")
    
    return joints

def visualize_mano_hands(processed_data, frame_idx=0, use_full_mano=True):
    """
    Visualize hands using proper MANO model integration.
    """
    
    # Extract data for the specified frame
    right_pos = processed_data['right_hand_position'][frame_idx]
    left_pos = processed_data['left_hand_position'][frame_idx]
    right_angles = processed_data['right_hand_angles'][frame_idx]
    left_angles = processed_data['left_hand_angles'][frame_idx]
    
    print(f"MANO-based visualization for frame {frame_idx}")
    
    # Convert MANO parameters to joint positions
    try:
        if use_full_mano:
            right_joints = convert_mano_params_to_joints(right_pos, right_angles, 'right')
            left_joints = convert_mano_params_to_joints(left_pos, left_angles, 'left')
        else:
            right_joints = approximate_joints_from_parameters(right_pos, right_angles, 'right')
            left_joints = approximate_joints_from_parameters(left_pos, left_angles, 'left')
    except Exception as e:
        print(f"Joint conversion failed: {e}")
        return
    
    # Create visualization
    fig = plt.figure(figsize=(15, 5))
    
    # MANO joint connections (21-joint model)
    mano_connections = [
        # Wrist to finger bases
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        # Thumb
        (1, 2), (2, 3), (3, 4),
        # Index
        (5, 6), (6, 7), (7, 8),
        # Middle  
        (9, 10), (10, 11), (11, 12),
        # Ring
        (13, 14), (14, 15), (15, 16),
        # Little
        (17, 18), (18, 19), (19, 20)
    ]
    
    # Plot right hand
    ax1 = fig.add_subplot(131, projection='3d')
    plot_hand_with_mano_structure(ax1, right_joints, mano_connections, 
                                 'Right Hand', 'red')
    
    # Plot left hand
    ax2 = fig.add_subplot(132, projection='3d')
    plot_hand_with_mano_structure(ax2, left_joints, mano_connections, 
                                 'Left Hand', 'blue')
    
    # Plot both hands together
    ax3 = fig.add_subplot(133, projection='3d')
    plot_hand_with_mano_structure(ax3, right_joints, mano_connections, 
                                 'Right', 'red', alpha=0.7)
    plot_hand_with_mano_structure(ax3, left_joints, mano_connections, 
                                 'Left', 'blue', alpha=0.7, add_to_existing=True)
    ax3.set_title('Both Hands')
    
    plt.suptitle(f'MANO-based Hand Visualization - Frame {frame_idx}')
    plt.tight_layout()
    plt.show()

def plot_hand_with_mano_structure(ax, joints, connections, title, color, alpha=1.0, add_to_existing=False):
    """
    Plot a hand with proper MANO joint structure.
    """
    
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
    
    # Highlight key joints
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
    
    # Set equal aspect ratio
    max_range = 0.1  # Adjust based on hand size
    ax.set_xlim([joints[0, 0] - max_range, joints[0, 0] + max_range])
    ax.set_ylim([joints[0, 1] - max_range, joints[0, 1] + max_range])
    ax.set_zlim([joints[0, 2] - max_range, joints[0, 2] + max_range])

def prepare_data_for_official_rendering(processed_data):
    """
    Prepare data in the format expected by official PianoMotion10M rendering.
    """
    
    # Get all frames
    num_frames = processed_data['num_frames']
    
    # Prepare data in official format
    right_data = np.concatenate([
        processed_data['right_hand_position'],  # (frames, 3)
        processed_data['right_hand_angles']     # (frames, 48)
    ], axis=1)  # Result: (frames, 51)
    
    left_data = np.concatenate([
        processed_data['left_hand_position'],   # (frames, 3)
        processed_data['left_hand_angles']      # (frames, 48)
    ], axis=1)  # Result: (frames, 51)
    
    print(f"Prepared data for official rendering:")
    print(f"  - Right hand data shape: {right_data.shape}")
    print(f"  - Left hand data shape: {left_data.shape}")
    print(f"  - Frame count: {num_frames}")
    print(f"  - Data format: [position(3), rotation_params(48)] per hand")
    
    return right_data, left_data

def use_official_rendering_if_available(right_data, left_data, audio_array):
    """
    Use the official PianoMotion10M rendering if available.
    """
    
    try:
        # Import official rendering function
        import sys
        import os
        sys.path.append('PianoMotion10M')
        from datasets.show import render_result
        
        # Create output directory
        output_dir = "rendered_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Use official rendering
        render_result(
            output_dir,
            audio_array,
            right_data,
            left_data,
            save_video=False  # Set to True if you want video output
        )
        
        print(f"✓ Used official PianoMotion10M rendering")
        print(f"✓ Output saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Official rendering failed: {e}")
        print("Falling back to custom visualization")
        return False

def complete_mano_integration_pipeline(pose_hat, guide, audio_wave, device='cpu'):
    """
    Complete pipeline with proper MANO integration.
    """
    
    # Explain the integration
    integration_info = explain_pianomotion_mano_integration()
    
    # Process with proper coordinate system
    processed_data = process_model_output_with_proper_coordinates(pose_hat, guide, device)
    
    # Try official rendering first
    right_data, left_data = prepare_data_for_official_rendering(processed_data)
    
    if use_official_rendering_if_available(right_data, left_data, audio_wave):
        print("✓ Used official PianoMotion10M rendering pipeline")
    else:
        print("Using custom MANO-based visualization")
        
        # Use custom MANO visualization
        try:
            visualize_mano_hands(processed_data, frame_idx=-1, use_full_mano=True)
        except Exception as e:
            print(f"Full MANO visualization failed: {e}")
            print("Using simplified approximation")
            visualize_mano_hands(processed_data, frame_idx=-1, use_full_mano=False)
    
    return processed_data, integration_info 