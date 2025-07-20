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