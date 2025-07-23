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
        './mano',
        'PianoMotion10M/PianoMotion10M/mano',  # Nested structure
        os.path.join(os.path.dirname(__file__), 'PianoMotion10M', 'mano'),  # Relative to current file
        os.path.join(os.getcwd(), 'PianoMotion10M', 'mano')  # Absolute from current directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if MANO_RIGHT.pkl exists in this directory
            mano_file = os.path.join(path, 'MANO_RIGHT.pkl')
            if os.path.exists(mano_file):
                print(f"Found MANO model at: {os.path.abspath(path)}")
                return path
    
    print("MANO model not found in any of the expected locations:")
    for path in possible_paths:
        print(f"  - {os.path.abspath(path)}")
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
    detected_convention = processed_data.get('coordinate_system', {}).get('convention', {})
    
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
    
    # Use detected coordinate system labels if available
    x_label = detected_convention.get('x_axis', 'X (Lateral)')
    y_label = detected_convention.get('y_axis', 'Y (Vertical)')
    z_label = detected_convention.get('z_axis', 'Z (Depth)')
    
    # Create 3D visualization with proper coordinate system
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Hand positions in world space
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(right_pos[0], right_pos[1], right_pos[2], 
               c='red', s=100, label='Right Hand', alpha=0.8)
    ax1.scatter(left_pos[0], left_pos[1], left_pos[2], 
               c='blue', s=100, label='Left Hand', alpha=0.8)
    
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_zlabel(z_label)
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
              color='red', arrow_length_ratio=0.1, label=x_label)
    # Y-axis (green)
    ax3.quiver(origin[0], origin[1], origin[2], 0, axes_length, 0, 
              color='green', arrow_length_ratio=0.1, label=y_label)
    # Z-axis (blue)
    ax3.quiver(origin[0], origin[1], origin[2], 0, 0, axes_length, 
              color='blue', arrow_length_ratio=0.1, label=z_label)
    
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
            "step_1": "Audio -> Position Predictor -> 3D hand positions",
            "step_2": "Audio + Positions -> Diffusion Model -> Hand rotation parameters",
            "step_3": "Positions + Rotations -> MANO Forward Kinematics -> Joint positions",
            "step_4": "Joint positions -> Visualization/Animation"
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
    
    print(f"[OK] Created approximate joints for {hand} hand ({len(joints)} joints)")
    
    return joints

def visualize_mano_hands(processed_data, frame_idx=0, use_full_mano=True):
    """
    Visualize hands using proper MANO model integration.
    """
    
    # Validate frame index
    num_frames = processed_data.get('num_frames', 0)
    if num_frames == 0:
        print("[ERROR] No frames available in processed data")
        return
    
    # Ensure frame_idx is valid
    if frame_idx < 0:
        frame_idx = 0  # Use first frame instead of negative index
        print(f"[WARNING] Negative frame index corrected to 0")
    elif frame_idx >= num_frames:
        frame_idx = num_frames - 1  # Use last frame
        print(f"[WARNING] Frame index {frame_idx} exceeds available frames, using last frame")
    
    # Extract data for the specified frame
    try:
        right_pos = processed_data['right_hand_position'][frame_idx]
        left_pos = processed_data['left_hand_position'][frame_idx]
        right_angles = processed_data['right_hand_angles'][frame_idx]
        left_angles = processed_data['left_hand_angles'][frame_idx]
    except (KeyError, IndexError) as e:
        print(f"[ERROR] Error accessing frame {frame_idx}: {e}")
        print(f"  Available data keys: {list(processed_data.keys())}")
        if 'num_frames' in processed_data:
            print(f"  Total frames: {processed_data['num_frames']}")
        return
    
    print(f"MANO-based visualization for frame {frame_idx} of {num_frames}")
    
    # Convert MANO parameters to joint positions
    try:
        if use_full_mano:
            right_joints = convert_mano_params_robust(right_pos, right_angles, 'right')
            left_joints = convert_mano_params_robust(left_pos, left_angles, 'left')
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
    
    plt.suptitle(f'MANO-based Hand Visualization - Frame {frame_idx} of {num_frames}')
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
        # Check for cv2 dependency first
        try:
            import cv2
        except ImportError:
            print("[ERROR] OpenCV (cv2) not available - required for official rendering")
            print("  Install with: pip install opencv-python")
            print("  Falling back to custom visualization")
            return False
        
        # Import official rendering function
        import sys
        import os
        sys.path.append('PianoMotion10M')
        
        try:
            from datasets.show import render_result
        except ImportError as e:
            print(f"[ERROR] Official rendering module not found: {e}")
            print("  This may be due to missing PianoMotion10M dependencies")
            print("  Falling back to custom visualization")
            return False
        
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
        
        print(f"[OK] Used official PianoMotion10M rendering")
        print(f"[OK] Output saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Official rendering failed: {e}")
        print("Falling back to custom visualization")
        return False

def complete_mano_integration_pipeline(pose_hat, guide, audio_wave, device='cpu', midi_path=None):
    """
    Complete pipeline with proper MANO integration and hand ordering verification.
    """
    
    # Explain the integration
    integration_info = explain_pianomotion_mano_integration()
    
    # Import the proper processing function from midi_to_frames.py
    try:
        import sys
        import os
        # Add the current directory to the path so we can import from midi_to_frames
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from midi_to_frames import process_model_output_with_proper_coordinates, verify_and_correct_hand_ordering
        print("[OK] Using proper coordinate processing from midi_to_frames.py")
    except ImportError as e:
        print(f"[WARNING] Could not import proper processing function: {e}")
        print("[WARNING] Using fallback processing (may not have correct scaling)")
        # Fall back to the local version
        processed_data = process_model_output_with_proper_coordinates(pose_hat, guide, device)
    else:
        # First, verify and correct hand ordering
        print("Step 3.1: Verifying hand ordering...")
        verified_guide, verified_pose, was_swapped = verify_and_correct_hand_ordering(
            guide, pose_hat, midi_path
        )
        
        # Use the proper processing function with coordinate transformations
        processed_data = process_model_output_with_proper_coordinates(verified_pose, verified_guide, device)
        
        # Add hand ordering metadata
        processed_data['hand_ordering_corrected'] = was_swapped
        if was_swapped:
            print("[OK] Hand ordering was corrected during processing")
        else:
            print("[OK] Hand ordering was verified as correct")
    
    # Try official rendering first
    right_data, left_data = prepare_data_for_official_rendering(processed_data)
    
    if use_official_rendering_if_available(right_data, left_data, audio_wave):
        print("[OK] Used official PianoMotion10M rendering pipeline")
    else:
        print("Using custom MANO-based visualization")
        
        # Use custom MANO visualization
        try:
            visualize_mano_hands(processed_data, frame_idx=0, use_full_mano=True)
        except Exception as e:
            print(f"Full MANO visualization failed: {e}")
            print("Using simplified approximation")
            visualize_mano_hands(processed_data, frame_idx=0, use_full_mano=False)
    
    return processed_data, integration_info 

def determine_mano_parameter_structure(sample_rotations):
    """Determine the correct MANO parameter structure through testing."""
    
    print("Determining MANO parameter structure...")
    
    # Based on official PianoMotion10M code analysis, we know the correct structure
    # Official rendering uses: position(3) + global_orient(3) + hand_pose(45) = 51 total
    # But our model outputs 48 rotation parameters, so we need to determine the structure
    
    # Test different interpretations
    test_position = np.array([0, 0.1, 0.2])  # Sample hand position
    
    # Test each interpretation
    best_interpretation, results = test_parameter_interpretations(test_position, sample_rotations)
    
    if best_interpretation:
        print(f"[OK] Detected parameter structure: {best_interpretation}")
        return best_interpretation
    else:
        print("[WARNING] Could not determine parameter structure, using fallback")
        return "standard_mano"  # Conservative fallback

def test_parameter_interpretations(hand_position, hand_rotations):
    """Test different ways to interpret the 48 rotation parameters."""
    
    interpretations = {
        "standard_mano": {
            "global_orient": hand_rotations[:3],
            "hand_pose": hand_rotations[3:48],
            "description": "Standard MANO: 3 global + 45 pose"
        },
        "all_pose": {
            "global_orient": np.zeros(3),
            "hand_pose": hand_rotations[:48],
            "description": "All pose: 0 global + 48 pose"
        },
        "extended_global": {
            "global_orient": hand_rotations[:6],
            "hand_pose": hand_rotations[6:48],
            "description": "Extended global: 6 global + 42 pose"
        },
        "pianomotion_style": {
            "global_orient": hand_rotations[:3],
            "hand_pose": hand_rotations[3:48],
            "description": "PianoMotion10M style: 3 global + 45 pose (but different interpretation)"
        }
    }
    
    results = {}
    
    for name, config in interpretations.items():
        try:
            print(f"\nTesting {name}: {config['description']}")
            
            # Try to create joint positions with this interpretation
            joints = convert_mano_params_with_structure(
                hand_position, 
                config['global_orient'], 
                config['hand_pose'],
                name
            )
            
            # Validate the resulting joint positions
            validation_score = validate_joint_positions(joints, name)
            
            results[name] = {
                'joints': joints,
                'validation_score': validation_score,
                'success': True
            }
            
            print(f"  [OK] Success: {name} produced valid joints (score: {validation_score:.2f})")
            
        except Exception as e:
            print(f"  [ERROR] Failed: {name} - {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Compare results
    successful_interpretations = {k: v for k, v in results.items() if v.get('success', False)}
    
    if successful_interpretations:
        best_interpretation = max(
            successful_interpretations.items(), 
            key=lambda x: x[1]['validation_score']
        )
        print(f"\n[BEST] Best interpretation: {best_interpretation[0]} (score: {best_interpretation[1]['validation_score']:.2f})")
        return best_interpretation[0], results
    else:
        print("\n[WARNING] No interpretation produced valid results")
        return None, results

def convert_mano_params_with_structure(hand_position, global_orient, hand_pose, method_name):
    """Convert MANO parameters to joint positions with specified structure."""
    
    print(f"  Converting with {method_name}:")
    print(f"    Global orient shape: {global_orient.shape if hasattr(global_orient, 'shape') else len(global_orient)}")
    print(f"    Hand pose shape: {hand_pose.shape if hasattr(hand_pose, 'shape') else len(hand_pose)}")
    
    # Use the appropriate conversion method based on parameter structure
    if len(global_orient) == 3 and len(hand_pose) == 45:
        # Standard MANO structure
        return convert_with_standard_mano(hand_position, global_orient, hand_pose)
    elif len(global_orient) == 0 and len(hand_pose) == 48:
        # All pose parameters
        return convert_with_all_pose(hand_position, hand_pose)
    else:
        # Custom structure - use approximation
        return convert_with_custom_structure(hand_position, global_orient, hand_pose)

def validate_joint_positions(joints, method_name):
    """Validate joint positions and return a quality score."""
    
    if joints is None or len(joints) == 0:
        return 0.0
    
    score = 0.0
    max_score = 6.0
    
    # Check 1: Reasonable number of joints (expecting 21)
    if len(joints) == 21:
        score += 1.0
        print(f"    [OK] Correct joint count: 21")
    else:
        print(f"    [WARNING] Unexpected joint count: {len(joints)}")
    
    # Check 2: Realistic joint positions (not NaN, not extreme values)
    if not np.any(np.isnan(joints)) and not np.any(np.isinf(joints)):
        score += 1.0
        print(f"    [OK] No NaN/Inf values")
    else:
        print(f"    [WARNING] Contains NaN/Inf values")
    
    # Check 3: Reasonable coordinate ranges
    joint_range = np.ptp(joints, axis=0)  # Range in each dimension
    if all(0.01 <= r <= 0.5 for r in joint_range):  # 1cm to 50cm range
        score += 1.0
        print(f"    [OK] Reasonable coordinate ranges: {joint_range}")
    else:
        print(f"    [WARNING] Extreme coordinate ranges: {joint_range}")
    
    # Check 4: Hand structure (fingers extend from wrist)
    wrist_pos = joints[0]
    finger_distances = [np.linalg.norm(joints[i] - wrist_pos) for i in [4, 8, 12, 16, 20]]  # Fingertips
    if all(0.05 <= d <= 0.15 for d in finger_distances):  # 5cm to 15cm from wrist
        score += 1.0
        print(f"    [OK] Realistic finger lengths: {finger_distances}")
    else:
        print(f"    [WARNING] Unrealistic finger lengths: {finger_distances}")
    
    # Check 5: Bone length consistency
    bone_lengths = calculate_bone_lengths_simple(joints)
    if bone_lengths and all(0.01 <= bl <= 0.08 for bl in bone_lengths):  # 1cm to 8cm bones
        score += 1.0
        print(f"    [OK] Realistic bone lengths")
    else:
        print(f"    [WARNING] Unrealistic bone lengths")
    
    # Check 6: Hand chirality (right hand should have thumb on correct side)
    thumb_direction = joints[4] - joints[0]  # Thumb tip - wrist
    if thumb_direction[0] > 0:  # Assuming right hand, thumb should be positive X
        score += 1.0
        print(f"    [OK] Correct hand chirality")
    else:
        print(f"    [WARNING] Incorrect hand chirality")
    
    final_score = score / max_score
    print(f"    Total score: {score}/{max_score} = {final_score:.2f}")
    
    return final_score

def calculate_bone_lengths_simple(joints):
    """Calculate simple bone lengths for validation."""
    
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)   # Little
    ]
    
    bone_lengths = []
    for parent, child in connections:
        if parent < len(joints) and child < len(joints):
            length = np.linalg.norm(joints[child] - joints[parent])
            bone_lengths.append(length)
    
    return bone_lengths

def convert_mano_params_robust(hand_position, hand_rotations, hand='right', structure=None):
    """Convert MANO parameters with robust structure detection."""
    
    if structure is None:
        structure = determine_mano_parameter_structure(hand_rotations)
    
    print(f"Converting MANO parameters using {structure} structure")
    
    try:
        if structure == "standard_mano":
            return convert_with_standard_mano(hand_position, hand_rotations[:3], hand_rotations[3:], hand)
        elif structure == "all_pose":
            return convert_with_all_pose(hand_position, hand_rotations, hand)
        elif structure == "pianomotion_style":
            return convert_with_pianomotion_style(hand_position, hand_rotations, hand)
        else:
            # Use the best available method
            return convert_with_adaptive_structure(hand_position, hand_rotations, hand)
            
    except Exception as e:
        print(f"MANO conversion failed with {structure}: {e}")
        print("Falling back to approximation method")
        return approximate_joints_from_parameters(hand_position, hand_rotations, hand)

def convert_with_standard_mano(hand_position, global_orient, hand_pose, hand='right'):
    """Convert using standard MANO structure (3 global + 45 pose)."""
    
    # Implementation for standard MANO structure
    # This uses the existing logic in mano_utils.py
    return convert_mano_params_to_joints_legacy(hand_position, 
                                       np.concatenate([global_orient, hand_pose]), 
                                       hand)

def convert_with_all_pose(hand_position, hand_rotations, hand='right'):
    """Convert using all-pose structure (48 pose parameters)."""
    
    # When all parameters are pose parameters, there's no global orientation
    # This means the hand orientation is encoded in the pose parameters
    
    # Use a modified version of the conversion
    global_orient = np.zeros(3)  # No global orientation
    hand_pose = hand_rotations    # All 48 are pose parameters
    
    # Use different joint mapping for 48 pose parameters
    return convert_with_extended_pose_params(hand_position, hand_pose, hand)

def convert_with_pianomotion_style(hand_position, hand_rotations, hand='right'):
    """Convert using PianoMotion10M style structure."""
    
    # Based on official PianoMotion10M rendering code analysis
    # The official code expects: position(3) + global_orient(3) + hand_pose(45) = 51
    # But our model outputs 48 rotation parameters, so we need to adapt
    
    # For 48 parameters, we'll use: global_orient(3) + hand_pose(45)
    global_orient = hand_rotations[:3]
    hand_pose = hand_rotations[3:48]
    
    # Use the standard MANO conversion but with proper parameter interpretation
    return convert_with_standard_mano(hand_position, global_orient, hand_pose, hand)

def convert_with_extended_pose_params(hand_position, pose_params, hand='right'):
    """Convert using extended pose parameters (48 instead of 45)."""
    
    # This suggests 16 joints × 3 parameters = 48
    # Instead of the standard 15 joints × 3 parameters = 45
    
    if len(pose_params) != 48:
        raise ValueError(f"Expected 48 pose parameters, got {len(pose_params)}")
    
    # Reshape to 16 joints × 3 parameters
    joint_rotations = pose_params.reshape(16, 3)
    
    # Create extended joint structure with 16+1=17 joints (including wrist)
    # Map to the standard 21-joint MANO structure
    
    joints = create_joints_from_extended_params(hand_position, joint_rotations, hand)
    
    return joints

def create_joints_from_extended_params(hand_position, joint_rotations, hand='right'):
    """Create joint positions from 16 joint rotation parameters."""
    
    # This is a more sophisticated approach that uses the rotation parameters
    # to compute joint positions through forward kinematics
    
    # For now, use a simplified approach
    # In practice, this would need proper MANO forward kinematics
    
    hand_sign = 1 if hand == 'right' else -1
    
    # Base joint structure (similar to existing approach)
    joints = np.zeros((21, 3))
    joints[0] = hand_position  # Wrist
    
    # Apply rotations to compute finger positions
    # This is a simplified implementation - full MANO would be more complex
    
    finger_base_offsets = [
        [hand_sign * 0.02, 0.01, 0.02],    # Thumb base
        [hand_sign * 0.02, 0.08, 0.01],    # Index base
        [0, 0.09, 0.005],                  # Middle base
        [hand_sign * -0.02, 0.085, 0],     # Ring base
        [hand_sign * -0.04, 0.075, -0.005] # Little base
    ]
    
    finger_joint_indices = [
        [1, 2, 3, 4],      # Thumb
        [5, 6, 7, 8],      # Index
        [9, 10, 11, 12],   # Middle
        [13, 14, 15, 16],  # Ring
        [17, 18, 19, 20]   # Little
    ]
    
    # For each finger, use the rotation parameters to compute joint positions
    for finger_idx, (base_offset, joint_indices) in enumerate(zip(finger_base_offsets, finger_joint_indices)):
        if finger_idx < len(joint_rotations):
            # Use rotation parameters for this finger
            finger_rots = joint_rotations[finger_idx * 3:(finger_idx + 1) * 3]
            
            # Compute finger joints with rotation influence
            for i, joint_idx in enumerate(joint_indices):
                if i == 0:
                    # Base joint
                    joints[joint_idx] = hand_position + base_offset
                else:
                    # Subsequent joints influenced by rotations
                    prev_joint = joints[joint_indices[i-1]]
                    
                    # Simple rotation influence (simplified)
                    if i-1 < len(finger_rots):
                        rotation_factor = finger_rots[i-1] * 0.05
                        offset = np.array([0, 0.025, 0]) * (1 + rotation_factor)
                        joints[joint_idx] = prev_joint + offset
    
    return joints

def convert_with_adaptive_structure(hand_position, hand_rotations, hand='right'):
    """Convert using adaptive structure detection."""
    
    # Try to determine the best structure automatically
    if len(hand_rotations) == 48:
        # Most likely PianoMotion10M style: 3 global + 45 pose
        return convert_with_pianomotion_style(hand_position, hand_rotations, hand)
    else:
        # Fallback to approximation
        return approximate_joints_from_parameters(hand_position, hand_rotations, hand)

def convert_mano_params_to_joints(hand_position, hand_rotations, hand_pose=None, hand='right'):
    """
    Convert MANO parameters to joint positions.
    
    Supports both old format (position + 48 rotation params) and new corrected format 
    (position + global_orient + hand_pose matching PianoMotion10M show.py).
    
    Args:
        hand_position: (3,) array - 3D translation of hand
        hand_rotations: (48,) array - MANO rotation parameters (old format)
                       OR (3,) array - global orientation (new format, requires hand_pose)
        hand_pose: (45,) array - hand pose parameters (new format only)
        hand: 'right' or 'left'
    
    Returns:
        joint_positions: (21, 3) array - 3D positions of MANO joints
    """
    
    # Check if we have the new corrected format (3 separate parameter arrays)
    if hand_pose is not None:
        # New format: translation + global_orient + hand_pose
        # This matches the PianoMotion10M show.py format exactly
        print(f"Using corrected MANO format: translation(3) + global_orient(3) + hand_pose(45)")
        return convert_mano_params_official_format(hand_position, hand_rotations, hand_pose, hand)
    else:
        # Old format: use the robust conversion system
        return convert_mano_params_robust(hand_position, hand_rotations, hand)

def convert_mano_params_official_format(translation, global_orient, hand_pose, hand='right'):
    """
    Convert MANO parameters using the exact format from PianoMotion10M show.py.
    
    This function replicates the parameter structure used in the official rendering code:
    - translation: (3,) - 3D position (camera_t in show.py)
    - global_orient: (3,) - global orientation (global_orient in show.py) 
    - hand_pose: (45,) - hand pose parameters (hand_pose in show.py)
    
    Args:
        translation: (3,) array - 3D translation
        global_orient: (3,) array - global orientation  
        hand_pose: (45,) array - hand pose parameters
        hand: 'right' or 'left'
    
    Returns:
        joints: (21, 3) array - 3D joint positions
    """
    try:
        import torch
        import numpy as np
        
        # Try to load the MANO model exactly as in show.py
        mano_model_path = find_mano_model_path()
        if mano_model_path is None:
            raise FileNotFoundError("MANO model not found")
        
        # Load MANO model
        from PianoMotion10M.models.mano import build_mano
        mano_layer = build_mano()
        
        # Convert arrays to tensors
        translation_tensor = torch.tensor(translation, dtype=torch.float32).unsqueeze(0)
        global_orient_tensor = torch.tensor(global_orient, dtype=torch.float32).unsqueeze(0)  
        hand_pose_tensor = torch.tensor(hand_pose, dtype=torch.float32).unsqueeze(0)
        betas = torch.zeros(1, 10, dtype=torch.float32)  # Zero shape parameters
        
        # Use the appropriate hand model
        hand_key = 'right'  # PianoMotion10M uses 'right' model for both hands in show.py
        
        # Run MANO forward pass (exactly as in show.py lines 96-99, 115-118)
        if hand == 'right':
            output = mano_layer[hand_key](
                global_orient=global_orient_tensor,
                hand_pose=hand_pose_tensor,
                betas=betas,
                transl=translation_tensor
            )
            vertices = output.vertices[0].detach().numpy()  # Shape: (778, 3)
        else:
            # For left hand, PianoMotion10M flips X coordinate (show.py line 120)
            output = mano_layer[hand_key](
                global_orient=global_orient_tensor,
                hand_pose=hand_pose_tensor,
                betas=betas,
                transl=translation_tensor
            )
            vertices = output.vertices[0].detach().numpy()  # Shape: (778, 3)
            vertices[:, 0] = -1 * vertices[:, 0]  # Flip X for left hand
        
        # Extract joint positions from vertices using MANO joint regressor
        # MANO outputs joints as part of the forward pass
        if hasattr(output, 'joints'):
            joints = output.joints[0].detach().numpy()  # Shape: (21, 3)
            if hand == 'left':
                joints[:, 0] = -1 * joints[:, 0]  # Flip X for left hand
        else:
            # Fallback: approximate joints from vertices (first 21 vertices are usually joint-related)
            joints = vertices[:21]  # Approximate joint positions
        
        print(f"[OK] Created official MANO joints for {hand} hand using PianoMotion10M format")
        return joints
        
    except Exception as e:
        print(f"Official MANO conversion failed: {e}")
        # Fallback to approximation method
        print(f"Falling back to approximation method...")
        
        # Combine parameters back to original format for fallback
        combined_params = np.concatenate([global_orient, hand_pose])  # 48 parameters
        return approximate_joints_from_parameters(translation, combined_params, hand)

def convert_mano_params_to_joints_legacy(hand_position, hand_rotations, hand='right'):
    """
    Legacy conversion function - the original implementation.
    This is kept for backward compatibility and testing.
    """
    
    # PianoMotion10M outputs 48 rotation parameters, not 21 joint positions
    # The MANO model is designed for mesh rendering, not joint extraction
    # For visualization purposes, we'll create a reasonable joint mapping
    
    if len(hand_rotations) != 48:
        print(f"[ERROR] Expected 48 rotation parameters, got {len(hand_rotations)}")
        print("  Falling back to simplified joint approximation")
        return approximate_joints_from_parameters(hand_position, hand_rotations, hand)
    
    print(f"Converting 48 MANO parameters to 21 joint positions for {hand} hand")
    
    # Parse MANO parameters according to PianoMotion10M format:
    # - First 3: Global orientation (root rotation)
    # - Next 45: Hand pose (15 joints × 3 rotations each)
    global_orient = hand_rotations[:3]  # First 3 parameters
    hand_pose = hand_rotations[3:48]    # Next 45 parameters (15 joints × 3)
    
    # Create a more sophisticated joint mapping based on the rotation parameters
    # This is an approximation that uses the rotation parameters to influence joint positions
    
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
    
    # Apply global orientation influence
    if len(global_orient) >= 3:
        # Use global orientation to rotate the entire hand
        global_rot_x = global_orient[0] * 0.1  # Scale down
        global_rot_y = global_orient[1] * 0.1
        global_rot_z = global_orient[2] * 0.1
        
        # Apply rotations around each axis
        for i in range(len(joint_offsets)):
            x, y, z = joint_offsets[i]
            
            # Rotate around X-axis
            y_new = y * np.cos(global_rot_x) - z * np.sin(global_rot_x)
            z_new = y * np.sin(global_rot_x) + z * np.cos(global_rot_x)
            y, z = y_new, z_new
            
            # Rotate around Y-axis
            x_new = x * np.cos(global_rot_y) + z * np.sin(global_rot_y)
            z_new = -x * np.sin(global_rot_y) + z * np.cos(global_rot_y)
            x, z = x_new, z_new
            
            # Rotate around Z-axis
            x_new = x * np.cos(global_rot_z) - y * np.sin(global_rot_z)
            y_new = x * np.sin(global_rot_z) + y * np.cos(global_rot_z)
            x, y = x_new, y_new
            
            joint_offsets[i] = [x, y, z]
    
    # Apply finger-specific rotations from hand pose parameters
    if len(hand_pose) >= 45:
        # Map hand pose parameters to finger rotations
        # Each finger has 3 joints with 3 rotation parameters each = 9 parameters per finger
        finger_params = hand_pose.reshape(5, 9)  # 5 fingers, 9 params each
        
        for finger_idx in range(5):
            if finger_idx < len(finger_params):
                finger_rot = finger_params[finger_idx]
                
                # Apply finger-specific rotations
                # This is a simplified mapping - in reality, MANO has more complex joint relationships
                if finger_idx == 0:  # Thumb
                    joint_indices = [1, 2, 3, 4]
                elif finger_idx == 1:  # Index
                    joint_indices = [5, 6, 7, 8]
                elif finger_idx == 2:  # Middle
                    joint_indices = [9, 10, 11, 12]
                elif finger_idx == 3:  # Ring
                    joint_indices = [13, 14, 15, 16]
                else:  # Little
                    joint_indices = [17, 18, 19, 20]
                
                # Apply rotation to finger joints
                for i, joint_idx in enumerate(joint_indices):
                    if joint_idx < len(joint_offsets) and i < 3:
                        rot_factor = finger_rot[i] * 0.05  # Scale down
                        x, y, z = joint_offsets[joint_idx]
                        
                        # Simple rotation around Y-axis (finger curl)
                        cos_r = np.cos(rot_factor)
                        sin_r = np.sin(rot_factor)
                        joint_offsets[joint_idx, 0] = x * cos_r - z * sin_r
                        joint_offsets[joint_idx, 2] = x * sin_r + z * cos_r
    
    # Translate to world position
    joints = joint_offsets + hand_position
    
    print(f"[OK] Created MANO-based joints for {hand} hand ({len(joints)} joints)")
    print(f"  - Used {len(global_orient)} global orientation parameters")
    print(f"  - Used {len(hand_pose)} hand pose parameters")
    
    return joints 

def convert_with_custom_structure(hand_position, global_orient, hand_pose, hand='right'):
    """Convert using custom structure - fallback for unknown parameter structures."""
    
    print(f"  Using custom structure conversion (fallback)")
    print(f"    Global orient: {len(global_orient)} parameters")
    print(f"    Hand pose: {len(hand_pose)} parameters")
    
    # Combine all parameters and use approximation
    all_params = np.concatenate([global_orient, hand_pose])
    return approximate_joints_from_parameters(hand_position, all_params, hand) 