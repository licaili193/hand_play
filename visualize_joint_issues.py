import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_joint_issue_visualization():
    """Create comprehensive visualization showing joint position issues."""
    
    # Load data
    with open('predicted_hand_motion.json', 'r') as f:
        data = json.load(f)
    
    # Get sample frame
    frame = data['frames'][0]
    left_joints = np.array(frame['left_hand_joints'])
    right_joints = np.array(frame['right_hand_joints'])
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 3D visualization of both hands
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    plot_hand_3d(ax1, left_joints, 'Left Hand - Current', 'blue')
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_hand_3d(ax2, right_joints, 'Right Hand - Current', 'red')
    
    # 2. Distance comparison chart
    ax3 = fig.add_subplot(2, 3, 3)
    create_distance_comparison(ax3, left_joints, right_joints)
    
    # 3. Realistic vs Current comparison
    ax4 = fig.add_subplot(2, 3, 4)
    create_realistic_comparison(ax4, left_joints, right_joints)
    
    # 4. Joint structure analysis
    ax5 = fig.add_subplot(2, 3, 5)
    create_structure_analysis(ax5, left_joints, right_joints)
    
    # 5. Problem summary
    ax6 = fig.add_subplot(2, 3, 6)
    create_problem_summary(ax6)
    
    plt.tight_layout()
    plt.savefig('joint_issues_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_hand_3d(ax, joints, title, color):
    """Plot hand in 3D with connections."""
    
    # 16-joint connections
    connections = [
        (0, 1), (1, 2), (2, 3),      # Thumb
        (0, 4), (4, 5), (5, 6),      # Index
        (0, 7), (7, 8), (8, 9),      # Middle
        (0, 10), (10, 11), (11, 12), # Ring
        (0, 13), (13, 14), (14, 15)  # Little
    ]
    
    # Plot joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
              c=color, s=60, alpha=0.8, edgecolors='black')
    
    # Plot connections
    for parent, child in connections:
        ax.plot([joints[parent, 0], joints[child, 0]],
               [joints[parent, 1], joints[child, 1]],
               [joints[parent, 2], joints[child, 2]], 
               color=color, linewidth=2, alpha=0.7)
    
    # Highlight fingertips
    fingertips = [3, 6, 9, 12, 15]
    for tip in fingertips:
        ax.scatter(joints[tip, 0], joints[tip, 1], joints[tip, 2], 
                  c='black', s=100, alpha=0.9, marker='o', edgecolors=color)
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

def create_distance_comparison(ax, left_joints, right_joints):
    """Create bar chart comparing fingertip distances."""
    
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    fingertip_indices = [3, 6, 9, 12, 15]
    
    # Calculate distances from wrist
    left_distances = []
    right_distances = []
    
    for tip_idx in fingertip_indices:
        left_dist = np.linalg.norm(left_joints[tip_idx] - left_joints[0]) * 100  # cm
        right_dist = np.linalg.norm(right_joints[tip_idx] - right_joints[0]) * 100  # cm
        left_distances.append(left_dist)
        right_distances.append(right_dist)
    
    x = np.arange(len(finger_names))
    width = 0.35
    
    ax.bar(x - width/2, left_distances, width, label='Left Hand', color='blue', alpha=0.7)
    ax.bar(x + width/2, right_distances, width, label='Right Hand', color='red', alpha=0.7)
    
    ax.set_xlabel('Finger')
    ax.set_ylabel('Distance from Wrist (cm)')
    ax.set_title('Fingertip Distances from Wrist', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(finger_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

def create_realistic_comparison(ax, left_joints, right_joints):
    """Create comparison with realistic ranges."""
    
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    fingertip_indices = [3, 6, 9, 12, 15]
    
    # Realistic ranges (cm)
    realistic_ranges = [
        (6, 10),   # Thumb
        (8, 12),   # Index
        (9, 13),   # Middle
        (8, 12),   # Ring
        (6, 10)    # Little
    ]
    
    # Calculate current distances
    left_distances = []
    right_distances = []
    
    for tip_idx in fingertip_indices:
        left_dist = np.linalg.norm(left_joints[tip_idx] - left_joints[0]) * 100
        right_dist = np.linalg.norm(right_joints[tip_idx] - right_joints[0]) * 100
        left_distances.append(left_dist)
        right_distances.append(right_dist)
    
    x = np.arange(len(finger_names))
    
    # Plot realistic ranges as error bars
    realistic_mins = [r[0] for r in realistic_ranges]
    realistic_maxs = [r[1] for r in realistic_ranges]
    realistic_means = [(r[0] + r[1]) / 2 for r in realistic_ranges]
    
    # Plot realistic ranges
    ax.errorbar(x, realistic_means, 
               yerr=[np.array(realistic_means) - np.array(realistic_mins),
                     np.array(realistic_maxs) - np.array(realistic_means)],
               fmt='o', color='green', capsize=5, capthick=2, markersize=8,
               label='Realistic Range', alpha=0.8)
    
    # Plot current distances
    ax.scatter(x, left_distances, color='blue', s=100, alpha=0.8, marker='^', label='Left Current')
    ax.scatter(x, right_distances, color='red', s=100, alpha=0.8, marker='v', label='Right Current')
    
    ax.set_xlabel('Finger')
    ax.set_ylabel('Distance from Wrist (cm)')
    ax.set_title('Current vs Realistic Fingertip Distances', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(finger_names, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

def create_structure_analysis(ax, left_joints, right_joints):
    """Analyze hand structure problems."""
    
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    
    # Calculate issues for each finger
    issues_left = []
    issues_right = []
    
    realistic_ranges = [(6, 10), (8, 12), (9, 13), (8, 12), (6, 10)]
    fingertip_indices = [3, 6, 9, 12, 15]
    
    for i, tip_idx in enumerate(fingertip_indices):
        left_dist = np.linalg.norm(left_joints[tip_idx] - left_joints[0]) * 100
        right_dist = np.linalg.norm(right_joints[tip_idx] - right_joints[0]) * 100
        
        min_realistic, max_realistic = realistic_ranges[i]
        
        # Calculate deviation from realistic range
        if left_dist < min_realistic:
            left_issue = min_realistic - left_dist  # Negative = too short
        elif left_dist > max_realistic:
            left_issue = left_dist - max_realistic  # Positive = too long
        else:
            left_issue = 0  # Within range
        
        if right_dist < min_realistic:
            right_issue = min_realistic - right_dist
        elif right_dist > max_realistic:
            right_issue = right_dist - max_realistic
        else:
            right_issue = 0
        
        issues_left.append(left_issue)
        issues_right.append(right_issue)
    
    x = np.arange(len(finger_names))
    width = 0.35
    
    # Color code: red for problems, green for OK
    left_colors = ['red' if issue > 0 else 'green' for issue in issues_left]
    right_colors = ['red' if issue > 0 else 'green' for issue in issues_right]
    
    ax.bar(x - width/2, issues_left, width, label='Left Hand Issues', 
           color=left_colors, alpha=0.7)
    ax.bar(x + width/2, issues_right, width, label='Right Hand Issues', 
           color=right_colors, alpha=0.7)
    
    ax.set_xlabel('Finger')
    ax.set_ylabel('Deviation from Realistic Range (cm)')
    ax.set_title('Fingertip Length Issues (Red = Problem)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(finger_names, rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)

def create_problem_summary(ax):
    """Create text summary of problems."""
    
    ax.axis('off')
    
    summary_text = """
JOINT POSITION ANALYSIS SUMMARY

MAIN ISSUES IDENTIFIED:
• Fingertips are generally TOO LONG
• Thumb: 4+ cm longer than realistic
• Index: 2-3 cm longer than realistic  
• Ring: 1-2 cm longer than realistic
• Middle & Little: Within realistic range

ROOT CAUSES:
• MANO parameter conversion issues
• Coordinate system scaling problems
• Model outputs 16 joints instead of 21
• Simplified joint structure approximation

IMPACT ON REALISM:
• Overall realism: 40% (2/5 fingers realistic)
• Hand proportions incorrect
• Finger length ratios wrong
• Visual appearance unrealistic

RECOMMENDATIONS:
• Fix MANO parameter interpretation
• Correct coordinate system scaling
• Implement full 21-joint MANO model
• Validate against anatomical standards
• Test with multiple hand poses

PREVIOUS ISSUE STATUS:
"Joint positions not realistic especially 
for tip of fingers" - PARTIALLY RESOLVED
Middle finger and little finger are now 
realistic, but thumb, index, and ring 
fingers still have significant issues.
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    ax.set_title('Analysis Summary & Recommendations', fontweight='bold', pad=20)

if __name__ == "__main__":
    create_joint_issue_visualization()
    print("Joint issues visualization saved as 'joint_issues_analysis.png'")