"""
Example script demonstrating 3D and 2D top-down view modes for hand gesture visualization.
"""

from visualize_hand_gestures import HandGestureVisualizer
import os

def main():
    """Demonstrate different view modes."""
    
    # Path to the predicted hand motion JSON file
    json_file = "predicted_hand_motion.json"
    
    # Check if the file exists
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found!")
        print("Please ensure the JSON file is in the current directory.")
        return
    
    print("=== Hand Gesture Visualization - View Mode Examples ===\n")
    
    # Example 1: 3D Visualization
    print("1. 3D Visualization Mode")
    print("-" * 30)
    print("Features:")
    print("• Full 3D perspective with depth")
    print("• Interactive rotation and zoom")
    print("• Shows all three dimensions (X, Y, Z)")
    print("• Best for understanding spatial relationships")
    print()
    
    visualizer_3d = HandGestureVisualizer(json_file, view_mode='3d')
    print(f"Loaded {visualizer_3d.num_frames} frames in 3D mode")
    
    # Show a sample frame in 3D
    print("Showing frame 10 in 3D mode...")
    visualizer_3d.visualize_single_frame(10)
    
    # Example 2: 2D Top-Down Visualization
    print("\n2. 2D Top-Down Visualization Mode")
    print("-" * 40)
    print("Features:")
    print("• Bird's eye view (X-Y plane)")
    print("• Simplified 2D representation")
    print("• Easier to see hand positioning")
    print("• Good for analyzing hand spread and finger positions")
    print("• Useful for piano key mapping analysis")
    print()
    
    visualizer_2d = HandGestureVisualizer(json_file, view_mode='2d_topdown')
    print(f"Loaded {visualizer_2d.num_frames} frames in 2D top-down mode")
    
    # Show a sample frame in 2D
    print("Showing frame 10 in 2D top-down mode...")
    visualizer_2d.visualize_single_frame(10)
    
    # Example 3: Interactive modes
    print("\n3. Interactive Visualization Modes")
    print("-" * 35)
    print("You can also use interactive mode with different views:")
    print()
    print("For 3D interactive mode:")
    print("  python visualize_hand_gestures.py predicted_hand_motion.json --mode interactive --view 3d")
    print()
    print("For 2D top-down interactive mode:")
    print("  python visualize_hand_gestures.py predicted_hand_motion.json --mode interactive --view 2d_topdown")
    print()
    
    # Example 4: Animation modes
    print("4. Animation Modes")
    print("-" * 20)
    print("Create animated GIFs with different views:")
    print()
    print("For 3D animation:")
    print("  python visualize_hand_gestures.py predicted_hand_motion.json --mode animation --view 3d --fps 15")
    print()
    print("For 2D top-down animation:")
    print("  python visualize_hand_gestures.py predicted_hand_motion.json --mode animation --view 2d_topdown --fps 15")
    print()
    
    # Show motion statistics for both modes
    print("5. Motion Statistics Comparison")
    print("-" * 35)
    
    stats_3d = visualizer_3d.calculate_motion_statistics()
    stats_2d = visualizer_2d.calculate_motion_statistics()
    
    print("Both view modes provide the same motion analysis:")
    print(f"  Total frames: {stats_3d['total_frames']}")
    print(f"  Duration: {stats_3d['duration_seconds']:.2f} seconds")
    print(f"  Left hand mean velocity: {stats_3d['left_hand_stats']['mean_velocity']:.4f}")
    print(f"  Right hand mean velocity: {stats_3d['right_hand_stats']['mean_velocity']:.4f}")
    print()
    
    print("=== View Mode Recommendations ===")
    print()
    print("Use 3D mode when:")
    print("• Analyzing hand depth and finger curling")
    print("• Understanding 3D spatial relationships")
    print("• Studying hand orientation and rotation")
    print("• General hand motion analysis")
    print()
    print("Use 2D top-down mode when:")
    print("• Analyzing hand positioning over piano keys")
    print("• Studying finger spread and reach")
    print("• Planning hand movements in 2D space")
    print("• Creating 2D motion diagrams")
    print()
    
    print("Both modes support:")
    print("• Uniform tick spacing and grids")
    print("• Interactive frame navigation")
    print("• Animation playback")
    print("• Export capabilities")
    print("• Motion statistics")
    
    print("\nVisualization examples completed!")

if __name__ == "__main__":
    main() 