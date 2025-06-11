import cv2
import numpy as np
import open3d as o3d
import torch # For YOLO
from ultralytics import YOLO # YOLOv8
import subprocess
import os
import sys
import time
import tempfile
import re 
import yaml
import json

# Assuming orbbec_camera.py is in 'camera' subdirectory
try:
    from camera.orbbec_camera import OrbbecCamera, get_serial_numbers
except ImportError as e:
    print(f"ERROR: Failed to import OrbbecCamera module: {e}")
    print("Please ensure 'camera' folder with 'orbbec_camera.py' is in the same directory or adjust PYTHONPATH.")
    sys.exit(1)

# --- USER CONFIGURATIONS ---
YOLO_MODEL_PATH = 'yolov8n.pt'  # Path to your YOLO model (e.g., yolov8n.pt, yolov8s.pt, or custom_model.pt)
# List of target class names that your YOLO model can detect AND you want to process
YOLO_TARGET_CLASS_NAMES = ['cup', 'bottle', 'person'] # IMPORTANT: Adjust to your model's classes and your targets
YOLO_CONF_THRESHOLD = 0.3  # Confidence threshold for YOLO detections

# IMPORTANT: Map detected class names to their corresponding CAD model files for Part 1
# The keys MUST match the names in YOLO_TARGET_CLASS_NAMES and your YOLO model output
CLASS_TO_MODEL_FILE_MAP = {
    'cup': 'path/to/your/cup_model.stl',       # Replace with actual path
    'bottle': 'path/to/your/bottle_model.ply', # Replace with actual path
    'person': 'path/to/your/person_model.obj'  # Example, replace with actual path if needed
}

# PART1_SCRIPT_NAME = "_estimate_pose_part1_pytorch_coarse_align.py" # No longer calling Part 1
PART2_SCRIPT_NAME = "_estimate_pose_part2_icp_estimation.py"
CONFIG_YAML_FILE = "pose_estimation_config.yaml"

ORCHESTRATOR_BASE_OUTPUT_DIR = "./yolo_orchestrator_direct_to_part2_runs"
MODEL_SAMPLE_POINTS_FOR_PART2 = 2048 # Points to sample from CAD model for Part2's target
# ---

def find_script_path(script_name):
    # (Copied from previous orchestrator, ensures scripts/configs can be found)
    path_in_cwd = os.path.join(os.getcwd(), script_name)
    if os.path.exists(path_in_cwd) and os.path.isfile(path_in_cwd):
        return os.path.abspath(path_in_cwd)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_in_script_dir = os.path.join(script_dir, script_name)
    if os.path.exists(path_in_script_dir) and os.path.isfile(path_in_script_dir):
        return os.path.abspath(path_in_script_dir)
    if script_name.endswith(".py"):
        print(f"Warning: Script {script_name} not found in standard locations. Assuming it's in PATH.")
        return script_name 
    print(f"ERROR: File {script_name} not found.")
    return None

def run_command_and_parse_output(command_list, parse_regex_dict, timeout_seconds=300):
    # (Copied from previous orchestrator)
    print(f"Executing command: {' '.join(command_list)}")
    results = {}
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        if process.returncode != 0:
            print(f"ERROR executing command: {' '.join(command_list)}")
            print(f"Return code: {process.returncode}")
            print("Stdout:"); print(stdout)
            print("Stderr:"); print(stderr)
            return None
        print("Command executed successfully. Stdout:"); print(stdout)
        if stderr: print("Stderr (warnings/minor errors):"); print(stderr)
        for key, pattern in parse_regex_dict.items():
            match = re.search(pattern, stdout)
            if match and match.group(1):
                results[key] = match.group(1).strip()
                print(f"  Parsed '{key}': {results[key]}")
            else:
                print(f"  Warning: Could not parse '{key}' from stdout using regex '{pattern}'.")
                results[key] = None
        return results
    except subprocess.TimeoutExpired:
        print(f"ERROR: Command timed out: {' '.join(command_list)}")
        if 'process' in locals() and process: process.kill(); process.communicate()
        return None
    except FileNotFoundError:
        print(f"ERROR: Command/script not found: {command_list[0]}. Check path and environment.")
        return None
    except Exception as e:
        print(f"ERROR running command {' '.join(command_list)}: {e}")
        return None

def crop_point_cloud_from_roi(full_pcd_o3d, roi_xyxy, color_cam_intrinsics_o3d, color_image_shape):
    """
    Crops an Open3D point cloud based on a 2D ROI in the color image.
    Assumes full_pcd_o3d points are in the color camera's 3D coordinate system.
    roi_xyxy: (xmin, ymin, xmax, ymax) for the bounding box.
    color_cam_intrinsics_o3d: Open3D PinholeCameraIntrinsic object for the color camera.
    color_image_shape: (height, width) of the color image.
    """
    if not full_pcd_o3d.has_points():
        print("Warning: Full point cloud is empty, cannot crop.")
        return o3d.geometry.PointCloud()

    points_np = np.asarray(full_pcd_o3d.points)
    colors_np = np.asarray(full_pcd_o3d.colors) if full_pcd_o3d.has_colors() else None

    # Project 3D points to 2D image plane
    # Open3D's project_points_to_image is not directly available. We do it manually.
    # Using pinhole camera model: u = (X * fx / Z) + cx, v = (Y * fy / Z) + cy
    fx = color_cam_intrinsics_o3d.intrinsic_matrix[0, 0]
    fy = color_cam_intrinsics_o3d.intrinsic_matrix[1, 1]
    cx = color_cam_intrinsics_o3d.intrinsic_matrix[0, 2]
    cy = color_cam_intrinsics_o3d.intrinsic_matrix[1, 2]

    X = points_np[:, 0]
    Y = points_np[:, 1]
    Z = points_np[:, 2]

    # Filter points in front of the camera
    valid_depth_mask = Z > 1e-3 # Small epsilon to avoid division by zero and points at camera origin
    if not np.any(valid_depth_mask):
        print("Warning: No points with valid depth in front of camera for projection.")
        return o3d.geometry.PointCloud()

    u = np.zeros_like(X)
    v = np.zeros_like(Y)

    u[valid_depth_mask] = (X[valid_depth_mask] * fx / Z[valid_depth_mask]) + cx
    v[valid_depth_mask] = (Y[valid_depth_mask] * fy / Z[valid_depth_mask]) + cy
    
    xmin, ymin, xmax, ymax = roi_xyxy
    
    # Create mask for points within ROI
    roi_mask = (u >= xmin) & (u < xmax) & (v >= ymin) & (v < ymax) & valid_depth_mask
    
    cropped_points = points_np[roi_mask]
    if cropped_points.shape[0] == 0:
        print("Warning: No points found within the specified ROI after projection.")
        return o3d.geometry.PointCloud()

    cropped_pcd_o3d = o3d.geometry.PointCloud()
    cropped_pcd_o3d.points = o3d.utility.Vector3dVector(cropped_points)

    if colors_np is not None and colors_np.shape[0] == points_np.shape[0]:
        cropped_colors = colors_np[roi_mask]
        cropped_pcd_o3d.colors = o3d.utility.Vector3dVector(cropped_colors)
    
    print(f"Cropped point cloud to {len(cropped_pcd_o3d.points)} points from ROI.")
    return cropped_pcd_o3d

def prepare_intermediate_data_for_part2(base_dir, object_pcd_o3d, cad_model_path, orchestrator_args_dict):
    """
    Prepares the directory and files that Part 2 expects, without running Part 1.
    Saves: 
    - args.json (from orchestrator_args_dict)
    - scene_observed_for_icp.pcd (the cropped object_pcd_o3d)
    - target_model_centered_for_icp.pcd (loaded from cad_model_path, sampled, centered)
    - common_target_model_original_scale.pcd
    - common_target_centroid_original_model_scale.npy
    - model_file_path.txt
    - initial_transform_for_icp.npy (identity matrix)
    Returns the path to this prepared intermediate directory.
    """
    os.makedirs(base_dir, exist_ok=True)
    print(f"Preparing intermediate data for Part 2 in: {base_dir}")

    # 1. Save args.json (can be minimal or based on orchestrator settings)
    args_file_path = os.path.join(base_dir, "args.json")
    try:
        with open(args_file_path, 'w') as f:
            json.dump(orchestrator_args_dict, f, indent=4)
        print(f"  Saved orchestrator args to {args_file_path}")
    except Exception as e:
        print(f"  ERROR saving args.json: {e}"); return None

    # 2. Save scene_observed_for_icp.pcd (the cropped object)
    path_scene_observed_for_icp = os.path.join(base_dir, "scene_observed_for_icp.pcd")
    try:
        o3d.io.write_point_cloud(path_scene_observed_for_icp, object_pcd_o3d)
        print(f"  Saved observed object (for ICP source) to: {path_scene_observed_for_icp}")
    except Exception as e:
        print(f"  ERROR saving scene_observed_for_icp.pcd: {e}"); return None

    # 3. Load, process, and save target CAD model files
    if not os.path.exists(cad_model_path):
        print(f"  ERROR: CAD model file for Part 2 target not found: {cad_model_path}"); return None
    
    target_pcd_original_model_scale_o3d = o3d.geometry.PointCloud()
    try:
        temp_mesh = o3d.io.read_triangle_mesh(cad_model_path)
        if temp_mesh.has_vertices():
            target_pcd_original_model_scale_o3d = temp_mesh.sample_points_uniformly(MODEL_SAMPLE_POINTS_FOR_PART2)
        else:
            target_pcd_original_model_scale_o3d = o3d.io.read_point_cloud(cad_model_path)
        if not target_pcd_original_model_scale_o3d.has_points():
            raise ValueError("CAD model has no points after loading/sampling for Part 2 target.")
        
        target_centroid_original_np = target_pcd_original_model_scale_o3d.get_center()
        target_pcd_centered_for_icp_o3d = o3d.geometry.PointCloud(target_pcd_original_model_scale_o3d)
        target_pcd_centered_for_icp_o3d.translate(-target_centroid_original_np)

        path_common_target_orig_scale = os.path.join(base_dir, "common_target_model_original_scale.pcd")
        path_common_target_centroid = os.path.join(base_dir, "common_target_centroid_original_model_scale.npy")
        path_model_file_txt = os.path.join(base_dir, "model_file_path.txt")
        path_target_centered_for_icp = os.path.join(base_dir, "target_model_centered_for_icp.pcd")

        o3d.io.write_point_cloud(path_common_target_orig_scale, target_pcd_original_model_scale_o3d)
        np.save(path_common_target_centroid, target_centroid_original_np)
        with open(path_model_file_txt, 'w') as f_model: f_model.write(cad_model_path)
        o3d.io.write_point_cloud(path_target_centered_for_icp, target_pcd_centered_for_icp_o3d)
        print(f"  Saved target model files (original, centroid, centered for ICP) to {base_dir}")

    except Exception as e_load_cad:
        print(f"  ERROR processing CAD model '{cad_model_path}' for Part 2: {e_load_cad}"); return None

    # 4. Save initial_transform_for_icp.npy (identity matrix as Part 1 is skipped)
    initial_transform_np = np.identity(4)
    path_initial_transform = os.path.join(base_dir, "initial_transform_for_icp.npy")
    try:
        np.save(path_initial_transform, initial_transform_np)
        print(f"  Saved identity initial transform to: {path_initial_transform}")
    except Exception as e_save_transform:
        print(f"  ERROR saving initial_transform_for_icp.npy: {e_save_transform}"); return None
    
    return base_dir # Return the path to the prepared directory

def main_yolo_orchestrator():
    print("YOLO Orchestrator Starting (Direct to Part 2 Mode)...")
    print(f"Attempting to load YOLO model from: {YOLO_MODEL_PATH}")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model from '{YOLO_MODEL_PATH}'. Error: {e}")
        print("Please ensure the model path is correct and Ultralytics YOLO is installed ('pip install ultralytics').")
        return

    print("Initializing Orbbec Camera...")
    camera_instance = None
    try:
        available_sns = get_serial_numbers()
        if not available_sns:
            print("ERROR: No Orbbec devices found. Please check connection."); return
        print(f"Found Orbbec devices: {available_sns}. Using first one: {available_sns[0]}")
        camera_instance = OrbbecCamera(available_sns[0])
        camera_instance.start_stream(depth_stream=True, color_stream=True, use_alignment=True, enable_sync=True)
        if camera_instance.param is None or camera_instance.param.rgb_intrinsic is None:
            raise RuntimeError("Failed to get RGB camera intrinsics from camera after starting stream.")
        print("Orbbec Camera initialized and stream started with D2C alignment.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Orbbec Camera or start stream: {e}")
        if camera_instance: camera_instance.stop()
        return

    os.makedirs(ORCHESTRATOR_BASE_OUTPUT_DIR, exist_ok=True)
    # part1_script_path = find_script_path(PART1_SCRIPT_NAME) # No longer needed
    part2_script_path = find_script_path(PART2_SCRIPT_NAME)
    config_yaml_path = find_script_path(CONFIG_YAML_FILE)

    if not all([part2_script_path, config_yaml_path]): # Check only Part2 script and config
        print("ERROR: Part 2 script or main config file not found. Exiting.")
        if camera_instance: camera_instance.stop(); return

    try:
        color_width = camera_instance.color_profile.get_width()
        color_height = camera_instance.color_profile.get_height()
        o3d_rgb_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=color_width, height=color_height,
            fx=camera_instance.param.rgb_intrinsic.fx, fy=camera_instance.param.rgb_intrinsic.fy,
            cx=camera_instance.param.rgb_intrinsic.cx, cy=camera_instance.param.rgb_intrinsic.cy
        )
        print("Successfully created Open3D RGB intrinsics object.")
    except Exception as e:
        print(f"ERROR: Could not get camera profile info or create O3D intrinsics: {e}")
        if camera_instance: camera_instance.stop(); return

    print("\nOrchestrator ready. Press 'r' in OpenCV window to detect objects and run pose estimation pipeline.")
    print("Press 'q' to quit.")

    # Create a dictionary of orchestrator settings that might be useful for args.json
    # This is a placeholder; Part 2 will load its own args from YAML and CLI overrides
    orchestrator_args_for_part2_json = {
        "yolo_model": YOLO_MODEL_PATH,
        "yolo_target_classes": YOLO_TARGET_CLASS_NAMES,
        "invoked_by_yolo_orchestrator_direct_to_part2": True
        # Add any other relevant info if Part 2 needs to know about its caller
    }

    try:
        while True:
            color_image, _, depth_frame_obj = camera_instance.get_frames()
            if color_image is None:
                cv2.imshow("YOLO Object Detection (Orbbec)", np.zeros((480,640,3), dtype=np.uint8))
                if cv2.waitKey(30) & 0xFF == ord('q'): break
                continue
            
            display_image = color_image.copy()
            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'): print("'q' pressed. Exiting..."); break
            
            if key == ord('r'):
                print("\n'r' pressed. Running YOLO detection and pose estimation pipeline (direct to Part 2)...")
                yolo_results = yolo_model(color_image, conf=YOLO_CONF_THRESHOLD)
                processed_one_object = False
                if yolo_results and len(yolo_results) > 0:
                    detections = yolo_results[0]
                    names = detections.names
                    print(f"YOLO found {len(detections.boxes)} objects in total.")
                    target_object_found_this_trigger = False

                    for i in range(len(detections.boxes)):
                        box = detections.boxes[i]
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = names.get(cls_id, f"ClassID_{cls_id}")
                        
                        if class_name in YOLO_TARGET_CLASS_NAMES and conf >= YOLO_CONF_THRESHOLD:
                            print(f"  Found target object: {class_name} with confidence {conf:.2f}")
                            target_object_found_this_trigger = True
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            cv2.rectangle(display_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                            cv2.putText(display_image, f"{class_name} {conf:.2f}", (xyxy[0], xyxy[1]-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                            
                            print("  Getting full point cloud for cropping...")
                            full_pcd_o3d = camera_instance.get_point_cloud(colored=True)
                            if full_pcd_o3d is None or not full_pcd_o3d.has_points():
                                print("  ERROR: Failed to get a valid full point cloud. Skipping this object."); continue

                            print(f"  Full point cloud has {len(full_pcd_o3d.points)} points.")
                            object_pcd_o3d = crop_point_cloud_from_roi(
                                full_pcd_o3d, xyxy, o3d_rgb_intrinsics, color_image.shape[:2]
                            )

                            if not object_pcd_o3d.has_points() or len(object_pcd_o3d.points) < 10:
                                print("  Warning: Cropped object point cloud is empty or too sparse. Skipping."); continue
                            
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            current_run_output_dir = os.path.join(ORCHESTRATOR_BASE_OUTPUT_DIR, f"run_{class_name}_{timestamp}")
                            # This will be the intermediate_dir for Part 2
                            intermediate_dir_for_part2 = os.path.join(current_run_output_dir, "intermediate_for_part2") 
                            
                            target_cad_model_file = CLASS_TO_MODEL_FILE_MAP.get(class_name)
                            if not target_cad_model_file or not os.path.exists(target_cad_model_file):
                                print(f"  ERROR: CAD Model for class '{class_name}' not found. Path: '{target_cad_model_file}'. Skipping."); continue
                            
                            # Prepare the intermediate directory for Part 2 manually
                            prepared_intermediate_path = prepare_intermediate_data_for_part2(
                                intermediate_dir_for_part2,
                                object_pcd_o3d,              # This is the cropped ROI point cloud
                                target_cad_model_file,
                                orchestrator_args_for_part2_json
                            )
                            if not prepared_intermediate_path:
                                print("  ERROR: Failed to prepare intermediate data for Part 2. Skipping."); continue
                            
                            # --- Run Part 2 (ICP Estimation) directly ---
                            print(f"\n  --- Running Part 2 for {class_name} (using prepared data: {prepared_intermediate_path}) ---")
                            part2_final_results_dir = os.path.join(current_run_output_dir, "part2_final_poses")
                            os.makedirs(part2_final_results_dir, exist_ok=True)
                            cmd_part2 = [
                                sys.executable, part2_script_path,
                                "--intermediate_dir", prepared_intermediate_path, # Use the manually prepared dir
                                "--config_file", config_yaml_path,
                                "--output_dir_part2", part2_final_results_dir,
                                "--visualize_pose", "True",
                                "--visualize_pose_in_scene", "True",
                                "--save_results", "True",
                            ]
                            parse_dict_part2 = {"npz_file": r"Saved PyTorch ICP poses to (.*\.npz)"} # Adjust if Part2's output changes
                            part2_results = run_command_and_parse_output(cmd_part2, parse_dict_part2)

                            if not part2_results or not part2_results.get("npz_file"):
                                print("  ERROR: Part 2 failed or could not parse NPZ file path."); continue
                            npz_file_path = part2_results["npz_file"]
                            if not os.path.exists(npz_file_path):
                                print(f"  ERROR: NPZ file reported by Part 2 does not exist: {npz_file_path}"); continue
                            
                            print(f"\n  --- Results for {class_name} from {npz_file_path} ---")
                            try:
                                estimated_poses_data = np.load(npz_file_path)
                                if not estimated_poses_data.files:
                                    print("  Warning: NPZ file is empty (no poses saved by Part 2).")
                                for instance_key in estimated_poses_data.files:
                                    pose_matrix = estimated_poses_data[instance_key]
                                    print(f"    Estimated Pose Matrix for '{instance_key}':\n{pose_matrix}")
                            except Exception as e_load_npz:
                                print(f"  ERROR loading or processing NPZ file {npz_file_path}: {e_load_npz}")
                            
                            print(f"  Pipeline finished for detected object: {class_name}")
                            processed_one_object = True
                            break 
                    
                    if not target_object_found_this_trigger:
                        print("No target objects (from YOLO_TARGET_CLASS_NAMES) found with sufficient confidence.")
                else:
                    print("YOLO did not return any results for this frame.")
            
            cv2.imshow("YOLO Object Detection (Orbbec)", display_image)

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting orchestrator...")
    finally:
        if camera_instance:
            print("Stopping Orbbec camera...")
            camera_instance.stop()
        cv2.destroyAllWindows()
        print("YOLO Orchestrator finished (Direct to Part 2 Mode).")

if __name__ == "__main__":
    if not os.path.exists(CONFIG_YAML_FILE):
        print(f"ERROR: Main config file '{CONFIG_YAML_FILE}' not found."); sys.exit(1)
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"ERROR: YOLO model file '{YOLO_MODEL_PATH}' not found."); sys.exit(1)
    missing_models = False
    for cn in YOLO_TARGET_CLASS_NAMES:
        if cn not in CLASS_TO_MODEL_FILE_MAP or not CLASS_TO_MODEL_FILE_MAP[cn] or not os.path.exists(CLASS_TO_MODEL_FILE_MAP[cn]):
            print(f"ERROR: CAD model file for class '{cn}' is not defined or does not exist.")
            print(f"  Expected path: {CLASS_TO_MODEL_FILE_MAP.get(cn, 'Not specified')}")
            missing_models = True
    if missing_models: print("Please update CLASS_TO_MODEL_FILE_MAP."); sys.exit(1)

    main_yolo_orchestrator() 