import sys
import os
import datetime
import copy
import numpy as np
import argparse
import time
import json # For loading args
import glob # For finding instance files
import yaml # Added for YAML support

# --- 导入 Open3D ---
try:
    import open3d as o3d
    from open3d.pipelines import registration as o3d_reg
    OPEN3D_AVAILABLE = True
except ImportError:
    print("FATAL Error: Open3D not found. Please install Open3D (pip install open3d).")
    sys.exit(1)

# --- 可视化函数 (从 Part 1 复制过来，确保一致性) ---
def visualize_single_pcd(pcd, window_name="Point Cloud", point_color=[0.5, 0.5, 0.5]):
    if not OPEN3D_AVAILABLE: print("Open3D not available, skipping visualization."); return
    if not pcd.has_points(): print(f"Skipping visualization for {window_name}: No points."); return
    pcd_vis = copy.deepcopy(pcd)
    if not pcd_vis.has_colors(): pcd_vis.paint_uniform_color(point_color)
    print(f"\nDisplaying Point Cloud: {window_name} (Points: {len(pcd_vis.points)})")
    print("(Close the Open3D window to continue script execution...)")
    try:
        o3d.visualization.draw_geometries([pcd_vis], window_name=window_name)
        print(f"Visualization window '{window_name}' closed.")
    except Exception as e:
        print(f"Error during visualization of '{window_name}': {e}")

def visualize_icp_alignment(source_pcd_transformed_model, instance_pcd_observed, window_name="ICP Alignment Result"):
    if not OPEN3D_AVAILABLE: print("Open3D not available, skipping visualization."); return
    if source_pcd_transformed_model.has_points() or instance_pcd_observed.has_points():
        instance_pcd_vis = copy.deepcopy(instance_pcd_observed)
        transformed_model_vis = copy.deepcopy(source_pcd_transformed_model)
        transformed_model_vis.paint_uniform_color([1, 0.706, 0])  # Yellow
        instance_pcd_vis.paint_uniform_color([0, 0.651, 0.929])  # Blue
        print(f"\nDisplaying ICP Alignment: {window_name}...")
        print("Yellow: Transformed Model Sampled Points | Blue: Observed Instance Points (preprocessed)")
        print("(Close the Open3D window to continue script execution...)")
        try:
            o3d.visualization.draw_geometries([transformed_model_vis, instance_pcd_vis], window_name=window_name)
            print(f"Alignment visualization window '{window_name}' closed.")
        except Exception as e:
            print(f"Error during visualization: {e}")

def visualize_transformed_model_in_scene(original_scene_pcd, target_model_geometry, estimated_pose,
                                         window_name="Transformed Model in Original Scene"):
    if not OPEN3D_AVAILABLE: print("Open3D not available, skipping visualization."); return
    scene_vis = copy.deepcopy(original_scene_pcd)
    model_vis = copy.deepcopy(target_model_geometry)
    model_vis.transform(estimated_pose)
    if not scene_vis.has_colors(): scene_vis.paint_uniform_color([0.8, 0.8, 0.8])
    if isinstance(model_vis, o3d.geometry.TriangleMesh):
        model_vis.paint_uniform_color([0, 1, 0])
        if not model_vis.has_vertex_normals(): model_vis.compute_vertex_normals()
    elif isinstance(model_vis, o3d.geometry.PointCloud):
        model_vis.paint_uniform_color([0, 1, 0])

    print(f"\nDisplaying: {window_name}")
    print("Scene Color (e.g., Gray/Original RGB): Original Full Scene | Green: Transformed Target Model")
    print("(Close the Open3D window to continue script execution...)")
    geometries_to_draw = []
    if scene_vis.has_points(): geometries_to_draw.append(scene_vis)
    else: print(f"Warning: Original scene for '{window_name}' has no points.")
    if (isinstance(model_vis, o3d.geometry.TriangleMesh) and model_vis.has_vertices()) or \
            (isinstance(model_vis, o3d.geometry.PointCloud) and model_vis.has_points()):
        geometries_to_draw.append(model_vis)
    else: print(f"Warning: Transformed model for '{window_name}' has no geometry, not drawing it.")
    if len(geometries_to_draw) > 0:
        try:
            o3d.visualization.draw_geometries(geometries_to_draw, window_name=window_name)
            print(f"Visualization window '{window_name}' closed.")
        except Exception as e:
            print(f"Error during visualization of '{window_name}': {e}")
    else: print(f"Not enough valid geometries to display for '{window_name}'.")

# Namespace class to convert dict to object for args
class ArgsNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return str(self.__dict__)

# --- Helper function to load config from YAML (copied from Part 1) ---
def load_config_from_yaml(config_path):
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            print(f"Successfully loaded configuration from {config_path}")
            return config_data
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing YAML file {config_path}: {e}. Using default script arguments.")
            return {}
        except Exception as e:
            print(f"Warning: Could not read YAML file {config_path}: {e}. Using default script arguments.")
            return {}
    else:
        # It's okay for config file to not exist for Part 2, CLI args are primary for some settings.
        # print(f"Info: YAML config file {config_path} not found. Using default script arguments or CLI overrides.")
        return {}

# --- Helper function to get value from config dict or use default (copied from Part 1) ---
def get_config_value(config_dict, section_name, key_name, default_value):
    if config_dict and section_name in config_dict and key_name in config_dict[section_name]:
        return config_dict[section_name][key_name]
    return default_value

def main(cli_args_part2):
    print(f"Starting Part 2 (ICP Estimation from Intermediate Data) at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Loading intermediate data from: {cli_args_part2.intermediate_dir}")

    if not os.path.isdir(cli_args_part2.intermediate_dir):
        print(f"FATAL Error: Intermediate directory not found: {cli_args_part2.intermediate_dir}")
        sys.exit(1)

    # Load original args from part 1
    args_file_path = os.path.join(cli_args_part2.intermediate_dir, "args.json")
    if not os.path.exists(args_file_path):
        print(f"FATAL Error: args.json not found in {cli_args_part2.intermediate_dir}")
        sys.exit(1)
    try:
        with open(args_file_path, 'r') as f:
            args_dict = json.load(f)
        args = ArgsNamespace(**args_dict) # Convert dict to namespace object
        print("Successfully loaded original arguments from args.json")
        print(f"Original arguments: {args}")
    except Exception as e:
        print(f"FATAL Error loading or parsing args.json: {e}")
        sys.exit(1)

    # Override visualization flags if specified in part 2 CLI args
    if cli_args_part2.visualize_pose is not None: args.visualize_pose = cli_args_part2.visualize_pose
    if cli_args_part2.visualize_pose_in_scene is not None: args.visualize_pose_in_scene = cli_args_part2.visualize_pose_in_scene
    if cli_args_part2.save_results is not None: args.save_results = cli_args_part2.save_results
    if cli_args_part2.output_dir_part2 is not None: 
        args.output_dir = cli_args_part2.output_dir_part2 # Override output dir for final poses
    else: # If not specified, use a subfolder within the intermediate_dir for part2 results
        args.output_dir = os.path.join(cli_args_part2.intermediate_dir, "part2_results")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Final poses will be saved to: {args.output_dir}")


    # --- Load Common Intermediate Data ---
    path_common_target_centered = os.path.join(cli_args_part2.intermediate_dir, "common_target_model_centered.pcd")
    path_common_target_original_scale = os.path.join(cli_args_part2.intermediate_dir, "common_target_model_original_scale.pcd")
    path_common_target_centroid = os.path.join(cli_args_part2.intermediate_dir, "common_target_centroid_original_model_scale.npy")
    path_common_original_scene = os.path.join(cli_args_part2.intermediate_dir, "common_original_scene.pcd")
    path_model_file_txt = os.path.join(cli_args_part2.intermediate_dir, "model_file_path.txt")

    try:
        print("Loading common intermediate data...")
        target_pcd_model_centered_for_icp = o3d.io.read_point_cloud(path_common_target_centered)
        target_pcd_original_model_scale = o3d.io.read_point_cloud(path_common_target_original_scale)
        target_centroid_original_model_scale = np.load(path_common_target_centroid)
        original_scene_o3d_pcd = o3d.io.read_point_cloud(path_common_original_scene)
        with open(path_model_file_txt, 'r') as f_model_path: loaded_model_file_path = f_model_path.read().strip()
        
        # Try to load original mesh for better visualization in scene (if it was a mesh)
        target_mesh_for_scene_viz = o3d.geometry.TriangleMesh() # Default to empty mesh
        if os.path.exists(loaded_model_file_path):
            try:
                temp_mesh_viz = o3d.io.read_triangle_mesh(loaded_model_file_path)
                if temp_mesh_viz.has_vertices():
                    target_mesh_for_scene_viz = temp_mesh_viz
                    print(f"Successfully loaded original target mesh from {loaded_model_file_path} for visualization.")
                else:
                    print(f"Original model file {loaded_model_file_path} is not a valid mesh, will use point cloud for scene viz.")
            except Exception as e_mesh_load:
                print(f"Could not load original model as mesh from {loaded_model_file_path}: {e_mesh_load}. Will use point cloud for scene viz.")
        else:
            print(f"Original model file path {loaded_model_file_path} not found. Will use sampled point cloud for scene viz.")

        if not target_pcd_model_centered_for_icp.has_points() or \
           not target_pcd_original_model_scale.has_points() or \
           not original_scene_o3d_pcd.has_points():
            raise ValueError("One or more common PCD files failed to load or are empty.")
        print("Common data loaded successfully.")
    except Exception as e:
        print(f"FATAL Error loading common intermediate data: {e}")
        sys.exit(1)

    # --- Find and Process Instance Data ---
    instance_preprocessed_files = sorted(glob.glob(os.path.join(cli_args_part2.intermediate_dir, "instance_*_preprocessed.pcd")))
    if not instance_preprocessed_files:
        print("No preprocessed instance files found in the intermediate directory. Exiting.")
        sys.exit(0)

    print(f"\nFound {len(instance_preprocessed_files)} instances to process.")
    estimated_poses = {}

    for inst_pcd_file in instance_preprocessed_files:
        try:
            basename = os.path.basename(inst_pcd_file)
            # Extract instance ID: instance_{inst_id}_preprocessed.pcd
            parts = basename.split('_')
            if len(parts) < 3 or parts[0] != 'instance' or parts[-1] != 'preprocessed.pcd':
                print(f"Skipping unrecognized file format: {basename}")
                continue
            inst_id_str = parts[1]
            try: inst_id = int(inst_id_str)
            except ValueError: print(f"Skipping file with non-integer instance ID: {basename}"); continue
            
            print(f"\nProcessing Instance ID: {inst_id}")

            path_inst_centroid = os.path.join(cli_args_part2.intermediate_dir, f"instance_{inst_id_str}_centroid.npy")
            path_inst_pca_transform = os.path.join(cli_args_part2.intermediate_dir, f"instance_{inst_id_str}_pca_transform.npy")

            if not os.path.exists(path_inst_centroid) or not os.path.exists(path_inst_pca_transform):
                print(f"  Missing centroid or PCA transform file for instance {inst_id}. Skipping.")
                continue

            instance_pcd_preprocessed = o3d.io.read_point_cloud(inst_pcd_file)
            instance_centroid_for_icp = np.load(path_inst_centroid)
            initial_transform_icp = np.load(path_inst_pca_transform)

            if not instance_pcd_preprocessed.has_points():
                print(f"  Instance {inst_id} preprocessed PCD is empty. Skipping.")
                continue
            print(f"  Loaded preprocessed instance PCD ({len(instance_pcd_preprocessed.points)} points), centroid, and PCA transform.")

            # Center the loaded instance PCD for ICP (as done in Part 1 before PCA)
            instance_pcd_centered_for_icp = copy.deepcopy(instance_pcd_preprocessed)
            instance_pcd_centered_for_icp.translate(-instance_centroid_for_icp)

            # --- Check min points for ICP (using Part 2's own args) ---
            if len(instance_pcd_centered_for_icp.points) < cli_args_part2.icp_min_points:
                print(f"  Skipping Inst {inst_id}: Not enough points for ICP ({len(instance_pcd_centered_for_icp.points)} < {cli_args_part2.icp_min_points}).")
                continue

            # --- Configure ICP Parameters (from cli_args_part2, which loaded from YAML/CLI for Part 2) ---
            threshold_icp = cli_args_part2.icp_threshold
            method_str_icp = cli_args_part2.icp_estimation_method

            estimation_method_icp = o3d_reg.TransformationEstimationPointToPoint()
            if method_str_icp.lower() == 'point_to_plane':
                normal_radius_icp = threshold_icp * 2.0
                normal_radius_icp = max(1e-3, normal_radius_icp)
                try:
                    print(f"    Estimating normals for instance (radius={normal_radius_icp:.4f})")
                    instance_pcd_centered_for_icp.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_icp, max_nn=30))
                    # Target model normals should have been estimated in part 1 if needed for PCA, but re-check for ICP.
                    if not target_pcd_model_centered_for_icp.has_normals():
                         print(f"    Estimating normals for target model (radius={normal_radius_icp:.4f})")
                         target_pcd_model_centered_for_icp.estimate_normals(
                             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_icp, max_nn=30))
                    
                    if instance_pcd_centered_for_icp.has_normals() and target_pcd_model_centered_for_icp.has_normals():
                        print("    Using PointToPlane ICP.")
                        estimation_method_icp = o3d_reg.TransformationEstimationPointToPlane()
                    else:
                        print("    Warning: Failed to estimate normals for ICP. Falling back to PointToPoint ICP.")
                        estimation_method_icp = o3d_reg.TransformationEstimationPointToPoint()
                except Exception as e_normal:
                    print(f"    Warning: Error estimating normals for ICP ({e_normal}). Falling back to PointToPoint ICP.")
                    estimation_method_icp = o3d_reg.TransformationEstimationPointToPoint()
            else:
                print("    Using PointToPoint ICP as per configuration.")

            criteria_icp = o3d_reg.ICPConvergenceCriteria(relative_fitness=cli_args_part2.icp_relative_fitness,
                                                          relative_rmse=cli_args_part2.icp_relative_rmse,
                                                          max_iteration=cli_args_part2.icp_max_iter)
            
            # --- Perform ICP ---
            print(f"    Running ICP with threshold: {threshold_icp:.4f}, max_iter: {cli_args_part2.icp_max_iter}")
            start_time_icp = time.perf_counter()
            
            reg_result = o3d_reg.registration_icp(instance_pcd_centered_for_icp, 
                                                  target_pcd_model_centered_for_icp,
                                                  threshold_icp, 
                                                  initial_transform_icp, # PCA transform from Part 1
                                                  estimation_method_icp, 
                                                  criteria_icp)
            
            end_time_icp = time.perf_counter()
            duration_icp = end_time_icp - start_time_icp

            T_centered_s_to_centered_t = reg_result.transformation
            # Final pose composition (instance centroid -> world) @ (centered_instance -> centered_model) @ (model_centroid -> world)
            T_world_to_instance_centroid = np.eye(4); T_world_to_instance_centroid[:3, 3] = instance_centroid_for_icp
            T_model_centroid_to_world = np.eye(4); T_model_centroid_to_world[:3, 3] = -target_centroid_original_model_scale
            final_estimated_pose = T_world_to_instance_centroid @ T_centered_s_to_centered_t @ T_model_centroid_to_world
            estimated_poses[f"instance_{inst_id_str}"] = final_estimated_pose # Use string ID for npz key

            print(f"  ICP for Inst {inst_id}:")
            print(f"    ICP Duration    : {duration_icp:.4f} seconds")
            print(f"    Fitness         : {reg_result.fitness:.6f}")
            print(f"    Inlier RMSE     : {reg_result.inlier_rmse:.6f}")
            print(f"    Correspondence Set: {len(reg_result.correspondence_set)} pairs")
            print(f"    Transformation (centered_instance to centered_model):\n{reg_result.transformation}")

            if args.visualize_pose:
                # For visualization, transform the *original scale model* by the *final pose*
                # and show it against the *preprocessed instance* (not centered)
                transformed_cad_pcd_for_viz = copy.deepcopy(target_pcd_original_model_scale)
                transformed_cad_pcd_for_viz.transform(final_estimated_pose) # Transform original model with full pose
                visualize_icp_alignment(transformed_cad_pcd_for_viz, instance_pcd_preprocessed,
                                        window_name=f"ICP Align - Inst {inst_id}")
            
            if args.visualize_pose_in_scene:
                model_geometry_for_scene_viz = target_mesh_for_scene_viz if target_mesh_for_scene_viz.has_vertices() else target_pcd_original_model_scale
                visualize_transformed_model_in_scene(original_scene_o3d_pcd, model_geometry_for_scene_viz,
                                                     final_estimated_pose,
                                                     window_name=f"Final CAD in Scene - Inst {inst_id}")
        except Exception as e_icp_inst:
            print(f"Error during ICP processing for instance {inst_id}: {e_icp_inst}")
            import traceback; traceback.print_exc()

    if args.save_results and estimated_poses:
        # Generate a unique timestamp for the output file name
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        # Get a base name for the results file (e.g., from input_h5 sample index or input_point_cloud_file name)
        if hasattr(args, 'input_point_cloud_file') and args.input_point_cloud_file is not None:
            base_name_for_output = os.path.splitext(os.path.basename(args.input_point_cloud_file))[0]
        elif hasattr(args, 'sample_index'):
            base_name_for_output = f"h5sample_{args.sample_index}"
        else:
            base_name_for_output = "unknown_input"

        pose_filename = os.path.join(args.output_dir, f"estimated_poses_{base_name_for_output}_{timestamp}.npz")
        try:
            np.savez(pose_filename, **estimated_poses) # estimated_poses already has keys like 'instance_0'
            print(f"\nSaved poses to {pose_filename}")
        except Exception as e_save:
            print(f"\nError saving poses: {e_save}")
    
    print(f"\nPart 2 (ICP Estimation) finished at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config_file', type=str, default='pose_estimation_config.yaml', help='Path to the YAML configuration file.')
    cli_args, remaining_argv = pre_parser.parse_known_args()
    config_data = load_config_from_yaml(cli_args.config_file) # Load config for Part 2 specific args

    parser = argparse.ArgumentParser(description='Part 2: ICP Pose Estimation from Intermediate Data.', parents=[pre_parser])
    parser.add_argument('--intermediate_dir', type=str, 
                        default=get_config_value(config_data, 'Part2ScriptConfig', 'intermediate_dir', ''), # Allow YAML to provide, but still check if None later if it was required
                        help='Directory containing intermediate data from Part 1 (_estimate_pose_part1_preprocessing_pca.py).')
    # Add optional arguments to override visualization/saving flags from args.json OR YAML
    parser.add_argument('--visualize_pose', action=argparse.BooleanOptionalAction, 
                        default=get_config_value(config_data, 'Part2ScriptConfig', 'visualize_pose', None), # Part 1's args.json will provide initial, YAML can override, CLI overrides all
                        help='Override: Visualize ICP alignment.')
    parser.add_argument('--visualize_pose_in_scene', action=argparse.BooleanOptionalAction, 
                        default=get_config_value(config_data, 'Part2ScriptConfig', 'visualize_pose_in_scene', None),
                        help='Override: Visualize final transformed model in scene.')
    parser.add_argument('--save_results', action=argparse.BooleanOptionalAction, 
                        default=get_config_value(config_data, 'Part2ScriptConfig', 'save_results', None),
                        help='Override: Save estimated poses to a .npz file.')
    parser.add_argument('--output_dir_part2', type=str, 
                        default=get_config_value(config_data, 'Part2ScriptConfig', 'output_dir_part2', None), 
                        help='Override output directory for final poses. If None, uses a subfolder in intermediate_dir.')

    # --- ICP Parameters (Now defined and used directly by Part 2) ---
    icp_params_group = parser.add_argument_group('ICP Parameters (from YAML or CLI for Part 2)')
    icp_params_group.add_argument('--icp_threshold', type=float, 
                                default=get_config_value(config_data, 'ICPParameters', 'icp_threshold', 20.0),
                                help='ICP max_correspondence_distance.')
    icp_params_group.add_argument('--icp_estimation_method', type=str, 
                                default=get_config_value(config_data, 'ICPParameters', 'icp_estimation_method', 'point_to_plane'), 
                                choices=['point_to_point', 'point_to_plane'], help="ICP estimation method.")
    icp_params_group.add_argument('--icp_relative_fitness', type=float, 
                                default=get_config_value(config_data, 'ICPParameters', 'icp_relative_fitness', 1e-8),
                                help='ICP convergence: relative fitness.')
    icp_params_group.add_argument('--icp_relative_rmse', type=float, 
                                default=get_config_value(config_data, 'ICPParameters', 'icp_relative_rmse', 1e-8),
                                help='ICP convergence: relative RMSE.')
    icp_params_group.add_argument('--icp_max_iter', type=int, 
                                default=get_config_value(config_data, 'ICPParameters', 'icp_max_iter', 20000),
                                help='ICP convergence: max iterations.')
    icp_params_group.add_argument('--icp_min_points', type=int, 
                                default=get_config_value(config_data, 'ICPParameters', 'icp_min_points', 100),
                                help='Min instance points required for ICP processing.')

    cli_args_part2 = parser.parse_args(remaining_argv) # Parse remaining args

    # Crucial check for intermediate_dir since it's fundamental
    if cli_args_part2.intermediate_dir is None:
        parser.error("the following arguments are required: --intermediate_dir (must be provided via CLI or YAML config)")

    main(cli_args_part2) 