import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_joint_realism():
    """Detailed analysis of joint positions for realism verification."""
    
    print("=" * 80)
    print("DETAILED JOINT POSITION ANALYSIS")
    print("Verification of Fingertip Realism in MANO Conversion")
    print("=" * 80)
    
    # Load data
    with open('predicted_hand_motion.json', 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    print(f"Total frames analyzed: {len(frames)}")
    
    # Analyze multiple frames for better statistics
    sample_frames = min(10, len(frames))
    all_distances = {'left': [], 'right': []}
    
    print(f"\nAnalyzing {sample_frames} frames for detailed statistics...")
    
    # 16-joint model fingertip indices
    fingertip_indices = [3, 6, 9, 12, 15]  # Thumb, Index, Middle, Ring, Little
    fingertip_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    
    for frame_idx in range(sample_frames):
        frame = frames[frame_idx]
        
        for hand_name, hand_key in [('left', 'left_hand_joints'), ('right', 'right_hand_joints')]:
            if hand_key in frame:
                joints = np.array(frame[hand_key])
                wrist_pos = joints[0]
                
                distances = []
                for tip_idx in fingertip_indices:
                    if tip_idx < len(joints):
                        tip_pos = joints[tip_idx]
                        distance = np.linalg.norm(tip_pos - wrist_pos)
                        distances.append(distance)
                
                all_distances[hand_name].append(distances)
    
    # Calculate comprehensive statistics
    print("\n" + "=" * 60)
    print("COMPREHENSIVE FINGERTIP DISTANCE ANALYSIS")
    print("=" * 60)
    
    for hand_name in ['left', 'right']:
        if all_distances[hand_name]:
            distances_array = np.array(all_distances[hand_name])  # Shape: (frames, 5)
            
            print(f"\n{hand_name.upper()} HAND DETAILED ANALYSIS:")
            print("-" * 40)
            
            for i, finger_name in enumerate(fingertip_names):
                finger_distances = distances_array[:, i]
                
                # Calculate detailed statistics
                mean_dist = np.mean(finger_distances)
                median_dist = np.median(finger_distances)
                std_dist = np.std(finger_distances)
                min_dist = np.min(finger_distances)
                max_dist = np.max(finger_distances)
                
                # Convert to centimeters for easier interpretation
                mean_cm = mean_dist * 100
                std_cm = std_dist * 100
                min_cm = min_dist * 100
                max_cm = max_dist * 100
                
                print(f"{finger_name:7}: Mean={mean_cm:5.1f}cm, Std={std_cm:4.1f}cm, Range=[{min_cm:5.1f}-{max_cm:5.1f}]cm")
    
    # Compare with realistic ranges
    print("\n" + "=" * 60)
    print("REALISM ASSESSMENT")
    print("=" * 60)
    
    # More nuanced realistic ranges (in centimeters)
    realistic_ranges_cm = {
        'Thumb': (6, 10),    # 6-10 cm from wrist to thumb tip
        'Index': (8, 12),    # 8-12 cm from wrist to index tip
        'Middle': (9, 13),   # 9-13 cm from wrist to middle tip (longest finger)
        'Ring': (8, 12),     # 8-12 cm from wrist to ring tip
        'Little': (6, 10)    # 6-10 cm from wrist to little tip
    }
    
    print("\nComparison with Realistic Human Hand Proportions:")
    print("(Based on adult hand anatomy literature)")
    print("-" * 50)
    
    overall_realism = {'left': 0, 'right': 0}
    total_fingers = len(fingertip_names)
    
    for hand_name in ['left', 'right']:
        if all_distances[hand_name]:
            distances_array = np.array(all_distances[hand_name])
            
            print(f"\n{hand_name.upper()} HAND REALISM CHECK:")
            realistic_count = 0
            
            for i, finger_name in enumerate(fingertip_names):
                finger_distances = distances_array[:, i]
                mean_dist_cm = np.mean(finger_distances) * 100
                
                min_realistic, max_realistic = realistic_ranges_cm[finger_name]
                is_realistic = min_realistic <= mean_dist_cm <= max_realistic
                
                status = "[REALISTIC]" if is_realistic else "[UNREALISTIC]"
                deviation = ""
                
                if not is_realistic:
                    if mean_dist_cm < min_realistic:
                        deviation = f" (TOO SHORT by {min_realistic - mean_dist_cm:.1f}cm)"
                    else:
                        deviation = f" (TOO LONG by {mean_dist_cm - max_realistic:.1f}cm)"
                
                print(f"  {finger_name:7}: {mean_dist_cm:5.1f}cm | Expected: {min_realistic:2d}-{max_realistic:2d}cm | {status}{deviation}")
                
                if is_realistic:
                    realistic_count += 1
            
            realism_percentage = (realistic_count / total_fingers) * 100
            overall_realism[hand_name] = realism_percentage
            
            print(f"  Overall Realism: {realistic_count}/{total_fingers} fingers ({realism_percentage:.0f}%)")
    
    # Analysis of coordinate ranges
    print("\n" + "=" * 60)
    print("3D COORDINATE ANALYSIS")
    print("=" * 60)
    
    for hand_name, hand_key in [('left', 'left_hand_joints'), ('right', 'right_hand_joints')]:
        print(f"\n{hand_name.upper()} HAND COORDINATE RANGES:")
        
        all_coords = []
        for frame_idx in range(sample_frames):
            frame = frames[frame_idx]
            if hand_key in frame:
                joints = np.array(frame[hand_key])
                all_coords.append(joints)
        
        if all_coords:
            all_coords = np.vstack(all_coords)  # Combine all joint positions
            
            x_range = np.ptp(all_coords[:, 0])  # Peak-to-peak (max - min)
            y_range = np.ptp(all_coords[:, 1])
            z_range = np.ptp(all_coords[:, 2])
            
            print(f"  X-axis range: {x_range:.4f}m ({x_range*100:.1f}cm)")
            print(f"  Y-axis range: {y_range:.4f}m ({y_range*100:.1f}cm)")
            print(f"  Z-axis range: {z_range:.4f}m ({z_range*100:.1f}cm)")
            
            # Check if ranges are reasonable for a hand
            reasonable_range = 0.05 <= x_range <= 0.25 and 0.1 <= y_range <= 0.3 and 0.02 <= z_range <= 0.15
            print(f"  Coordinate ranges reasonable: {'YES' if reasonable_range else 'NO'}")
    
    # Finger proportion analysis
    print("\n" + "=" * 60)
    print("FINGER PROPORTION ANALYSIS")
    print("=" * 60)
    
    print("\nAnalyzing relative finger lengths (proportions):")
    print("Typical proportions: Middle > Index ~ Ring > Thumb > Little")
    
    for hand_name in ['left', 'right']:
        if all_distances[hand_name]:
            distances_array = np.array(all_distances[hand_name])
            mean_distances = np.mean(distances_array, axis=0)
            
            # Sort fingers by length
            finger_lengths = list(zip(fingertip_names, mean_distances * 100))
            finger_lengths.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n{hand_name.upper()} HAND - Fingers ranked by length:")
            for rank, (finger, length_cm) in enumerate(finger_lengths, 1):
                print(f"  {rank}. {finger:7}: {length_cm:5.1f}cm")
            
            # Check if proportions follow typical pattern
            middle_idx = fingertip_names.index('Middle')
            index_idx = fingertip_names.index('Index')
            ring_idx = fingertip_names.index('Ring')
            thumb_idx = fingertip_names.index('Thumb')
            little_idx = fingertip_names.index('Little')
            
            middle_len = mean_distances[middle_idx]
            index_len = mean_distances[index_idx]
            ring_len = mean_distances[ring_idx]
            thumb_len = mean_distances[thumb_idx]
            little_len = mean_distances[little_idx]
            
            # Check typical relationships
            checks = {
                'Middle longest': middle_len == max(mean_distances),
                'Little shortest': little_len == min(mean_distances),
                'Index > Thumb': index_len > thumb_len,
                'Ring > Thumb': ring_len > thumb_len,
                'Middle > Index': middle_len > index_len,
                'Middle > Ring': middle_len > ring_len
            }
            
            print(f"  Proportion checks:")
            for check_name, passed in checks.items():
                status = "PASS" if passed else "FAIL"
                print(f"    {check_name:15}: {status}")
            
            proportion_score = sum(checks.values()) / len(checks) * 100
            print(f"  Proportion accuracy: {proportion_score:.0f}%")
    
    # Final assessment
    print("\n" + "=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)
    
    left_realism = overall_realism.get('left', 0)
    right_realism = overall_realism.get('right', 0)
    average_realism = (left_realism + right_realism) / 2
    
    print(f"\nFINGERTIP REALISM SCORES:")
    print(f"  Left hand:  {left_realism:.0f}% realistic")
    print(f"  Right hand: {right_realism:.0f}% realistic")
    print(f"  Average:    {average_realism:.0f}% realistic")
    
    if average_realism >= 80:
        assessment = "EXCELLENT - Joint positions are highly realistic"
    elif average_realism >= 60:
        assessment = "GOOD - Joint positions are mostly realistic with minor issues"
    elif average_realism >= 40:
        assessment = "FAIR - Joint positions have some realism issues"
    else:
        assessment = "POOR - Joint positions have significant realism issues"
    
    print(f"\nOVERALL ASSESSMENT: {assessment}")
    
    # Issues and recommendations
    print(f"\nKEY FINDINGS:")
    
    for hand_name in ['left', 'right']:
        if all_distances[hand_name]:
            distances_array = np.array(all_distances[hand_name])
            
            issues = []
            for i, finger_name in enumerate(fingertip_names):
                mean_dist_cm = np.mean(distances_array[:, i]) * 100
                min_realistic, max_realistic = realistic_ranges_cm[finger_name]
                
                if mean_dist_cm > max_realistic:
                    issues.append(f"{finger_name} too long ({mean_dist_cm:.1f}cm vs max {max_realistic}cm)")
                elif mean_dist_cm < min_realistic:
                    issues.append(f"{finger_name} too short ({mean_dist_cm:.1f}cm vs min {min_realistic}cm)")
            
            if issues:
                print(f"  {hand_name.upper()} hand issues:")
                for issue in issues:
                    print(f"    - {issue}")
            else:
                print(f"  {hand_name.upper()} hand: All fingertips within realistic ranges")
    
    # Compare with previous issues
    print(f"\nCOMPARISON WITH PREVIOUS ANALYSIS:")
    print(f"  Previous issue: 'Joint positions are not realistic especially for the tip of fingers'")
    
    if average_realism >= 60:
        print(f"  IMPROVEMENT: Fingertip positions now show {average_realism:.0f}% realism")
        print(f"  STATUS: Significant improvement in joint positioning")
    else:
        print(f"  CONCERN: Fingertip positions still show only {average_realism:.0f}% realism")
        print(f"  STATUS: Further improvements needed in MANO conversion")
    
    return {
        'left_realism': left_realism,
        'right_realism': right_realism,
        'average_realism': average_realism,
        'assessment': assessment
    }

if __name__ == "__main__":
    results = analyze_joint_realism()
    print(f"\nAnalysis complete. Results saved to console output.")