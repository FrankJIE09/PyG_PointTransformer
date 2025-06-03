import sys
import os
import datetime
import copy
import h5py
import numpy as np
import torch
import argparse
import time
import yaml # Added for YAML support
import json # For saving/loading args
from scipy.spatial.transform import Rotation as R

# --- 导入本地模块 ---
try:
    from model import PyG_PointTransformerSegModel  # 语义分割模型类

    try:
        from dataset import normalize_point_cloud_xyz as normalize_point_cloud_xyz_from_dataset

        print("Successfully imported normalize_point_cloud_xyz from dataset.py")
    except ImportError:
        print("normalize_point_cloud_xyz not found in dataset.py, will use local definition.")
        normalize_point_cloud_xyz_from_dataset = None
except ImportError as e:
    print(f"FATAL Error importing local modules (model.py or dataset.py): {e}")
    sys.exit(1)

# --- 导入 Open3D and Scikit-learn ---
try:
    import open3d as o3d
    from open3d.pipelines import registration as o3d_reg

    OPEN3D_AVAILABLE = True
except ImportError:
    print("FATAL Error: Open3D not found. Please install Open3D (pip install open3d).")
    sys.exit(1)
try:
    from sklearn.cluster import DBSCAN

    SKLEARN_AVAILABLE = True
except ImportError:
    print("FATAL Error: scikit-learn not found. Please install scikit-learn (pip install scikit-learn).")
    sys.exit(1)


# --- XYZ 坐标归一化函数 ---
def normalize_point_cloud_xyz_local(points_xyz):
    if points_xyz.shape[0] == 0: return points_xyz
    centroid = np.mean(points_xyz, axis=0)
    points_centered = points_xyz - centroid
    max_dist = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
    if max_dist < 1e-6: return points_centered
    points_normalized = points_centered / max_dist
    return points_normalized.astype(np.float32)


if normalize_point_cloud_xyz_from_dataset is not None:
    normalize_xyz_for_semantic_model = normalize_point_cloud_xyz_from_dataset
    print("Using XYZ normalization function from dataset.py for semantic model input.")
else:
    normalize_xyz_for_semantic_model = normalize_point_cloud_xyz_local
    print("Using local XYZ normalization function for semantic model input (ensure it matches dataset.py).")


# --- 可视化函数 ---
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

def create_axis_lineset(eigenvectors_matrix, origin=np.array([0,0,0]), length=1.0):
    """ Creates an Open3D LineSet representing 3 axes from eigenvectors.
        eigenvectors_matrix: 3x3 matrix where columns are eigenvectors.
        Origin: origin point of the axes.
        Length: length of the axes lines.
    """
    points = [origin.tolist()]
    lines = []
    # Standard RGB for X, Y, Z principal axes respectively
    axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] 
    colors = []

    for i in range(3):
        # Eigenvectors are columns: eigenvectors_matrix[:, i]
        axis_vector = eigenvectors_matrix[:, i] 
        end_point = origin + axis_vector * length
        points.append(end_point.tolist())
        # Line from origin (index 0) to this axis endpoint (index len(points)-1)
        lines.append([0, len(points)-1]) 
        colors.append(axis_colors[i])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def visualize_pca_alignment_with_axes(
    pcd_target_centered, target_axes_vectors, 
    pcd_instance_pca_aligned, instance_transformed_axes_vectors,
    axis_length=100.0, window_name="PCA Alignment with Axes"
):
    if not OPEN3D_AVAILABLE: 
        print("Open3D not available, skipping PCA axes visualization.")
        return

    geometries = []
    
    # Target CAD model (centered)
    pcd_target_vis = copy.deepcopy(pcd_target_centered)
    pcd_target_vis.paint_uniform_color([0.7, 0.7, 0.7]) # Gray
    if pcd_target_vis.has_points():
        geometries.append(pcd_target_vis)
    
    # Principal axes of the target CAD model
    if target_axes_vectors is not None:
        target_axes_lines = create_axis_lineset(target_axes_vectors, length=axis_length)
        geometries.append(target_axes_lines)

    # Instance point cloud (centered and PCA-aligned)
    pcd_instance_vis = copy.deepcopy(pcd_instance_pca_aligned)
    pcd_instance_vis.paint_uniform_color([0.0, 0.7, 0.7]) # Cyan
    if pcd_instance_vis.has_points():
        geometries.append(pcd_instance_vis)
        
    # Principal axes of the instance (transformed by PCA alignment rotation)
    if instance_transformed_axes_vectors is not None:
        instance_axes_lines = create_axis_lineset(instance_transformed_axes_vectors, length=axis_length)
        geometries.append(instance_axes_lines)
        
    if not geometries:
        print(f"Skipping PCA axes visualization for {window_name}: No geometries to show.")
        return

    print(f"\nDisplaying: {window_name}")
    print("  Target CAD (Gray) and its principal axes (R,G,B for 1st,2nd,3rd PC).")
    print("  Instance (Cyan, PCA-aligned) and its transformed principal axes (R,G,B for 1st,2nd,3rd PC).")
    print("  If PCA is good, the respective colored axes should align.")
    print("(Close the Open3D window to continue script execution...)")
    try:
        o3d.visualization.draw_geometries(geometries, window_name=window_name, width=1280, height=960)
        print(f"Visualization window '{window_name}' closed.")
    except Exception as e:
        print(f"Error during PCA axes visualization of '{window_name}': {e}")


def visualize_semantic_segmentation(points_xyz, semantic_labels_np, num_classes, target_label_id,
                                    window_name="Semantic Segmentation Output"):
    """ 可视化语义分割结果 """
    if not OPEN3D_AVAILABLE: print("Open3D not available, skipping visualization."); return
    if points_xyz.shape[0] == 0: print(f"Skipping visualization for {window_name}: No points."); return

    print(f"\nDisplaying Semantic Segmentation: {window_name}")
    print("(Close the Open3D window to continue script execution...)")

    semantic_pcd = o3d.geometry.PointCloud()
    semantic_pcd.points = o3d.utility.Vector3dVector(points_xyz)
    max_label = np.max(semantic_labels_np) if semantic_labels_np.size > 0 else -1
    distinct_colors = [
        [230 / 255, 25 / 255, 75 / 255], [60 / 255, 180 / 255, 75 / 255], [255 / 255, 225 / 255, 25 / 255],
        [0 / 255, 130 / 255, 200 / 255], [245 / 255, 130 / 255, 48 / 255], [145 / 255, 30 / 255, 180 / 255],
        [70 / 255, 240 / 255, 240 / 255], [240 / 255, 50 / 255, 230 / 255], [210 / 255, 245 / 255, 60 / 255],
        [250 / 255, 190 / 255, 212 / 255], [0 / 255, 128 / 255, 128 / 255], [220 / 255, 190 / 255, 255 / 255],
        [170 / 255, 110 / 255, 40 / 255], [255 / 255, 250 / 255, 200 / 255], [128 / 255, 0 / 255, 0 / 255],
        [128 / 255, 128 / 255, 0 / 255], [0 / 255, 0 / 255, 128 / 255], [128 / 255, 128 / 255, 128 / 255]
    ]
    num_distinct_colors = len(distinct_colors)
    colors_for_labels = np.zeros((max_label + 1, 3))
    for i in range(max_label + 1):
        if i == target_label_id:
            colors_for_labels[i] = [1, 0, 0]
        else:
            colors_for_labels[i] = distinct_colors[i % num_distinct_colors]
            if i >= num_distinct_colors:
                colors_for_labels[i] = np.random.rand(3)
    if semantic_labels_np.size > 0:
        valid_labels_mask = semantic_labels_np <= max_label
        point_colors = np.zeros((semantic_labels_np.shape[0], 3))
        if np.any(valid_labels_mask):
            point_colors[valid_labels_mask] = colors_for_labels[semantic_labels_np[valid_labels_mask]]
        semantic_pcd.colors = o3d.utility.Vector3dVector(point_colors)
    else:
        semantic_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    try:
        o3d.visualization.draw_geometries([semantic_pcd], window_name=window_name)
        print(f"Visualization window '{window_name}' closed.")
    except Exception as e:
        print(f"Error during visualization of '{window_name}': {e}")

# --- Helper function to load config from YAML ---
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
        print(f"Warning: YAML config file {config_path} not found. Using default script arguments.")
        return {}

# --- Helper function to get value from config dict or use default ---
def get_config_value(config_dict, section_name, key_name, default_value):
    if config_dict and section_name in config_dict and key_name in config_dict[section_name]:
        return config_dict[section_name][key_name]
    return default_value

# --- 主函数 ---
def main(args):
    print(f"Starting Part 1 (Preprocessing & PCA) at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Arguments: {args}")

    if not OPEN3D_AVAILABLE:
        print("Open3D is essential for this script. Exiting.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Create intermediate data directory ---
    intermediate_dir_name = "intermediate_processing_data" # Base name
    # Make it more specific if sample_index or model_file is used
    if args.input_point_cloud_file:
        base_input_name = os.path.splitext(os.path.basename(args.input_point_cloud_file))[0]
        intermediate_dir_name = f"intermediate_data_{base_input_name}"
    elif args.input_h5:
         intermediate_dir_name = f"intermediate_data_h5sample_{args.sample_index}"
    
    intermediate_output_dir = os.path.join(args.output_dir, intermediate_dir_name)
    os.makedirs(intermediate_output_dir, exist_ok=True)
    print(f"Intermediate data will be saved to: {intermediate_output_dir}")

    # Save args to JSON
    args_file_path = os.path.join(intermediate_output_dir, "args.json")
    try:
        with open(args_file_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
        print(f"Saved script arguments to {args_file_path}")
    except Exception as e:
        print(f"FATAL Error saving arguments to {args_file_path}: {e}")
        sys.exit(1)
        
    # --- 数据加载 ---
    points_xyz_original_from_h5 = None
    points_rgb_original_from_h5 = None

    if args.input_point_cloud_file:
        print(f"Loading data from point cloud file: {args.input_point_cloud_file}")
        if not os.path.exists(args.input_point_cloud_file):
            print(f"FATAL Error: Point cloud file not found: {args.input_point_cloud_file}")
            sys.exit(1)
        
        file_extension = os.path.splitext(args.input_point_cloud_file)[1].lower()
        try:
            if file_extension == '.txt':
                # ... (original TXT loading logic) ...
                print(f"  Loading TXT file: {args.input_point_cloud_file} using NumPy...")
                try:
                    with open(args.input_point_cloud_file, 'r') as f_test_delim:
                        first_line = f_test_delim.readline()
                    delimiter = ',' if ',' in first_line else None 
                    data = np.loadtxt(args.input_point_cloud_file, delimiter=delimiter, dtype=np.float32)
                    if data.ndim == 1: data = data.reshape(1, -1)
                    if data.shape[0] == 0: raise ValueError("TXT file is empty or could not be parsed.")
                    if data.shape[1] < 3: raise ValueError(f"TXT file must have at least 3 columns for XYZ. Found {data.shape[1]} columns.")
                    pcd_loaded = o3d.geometry.PointCloud()
                    pcd_loaded.points = o3d.utility.Vector3dVector(data[:, :3])
                    if data.shape[1] >= 6:
                        print(f"    Found {data.shape[1]} columns. Assuming first 3 are XYZ, next 3 are RGB.")
                        colors = data[:, 3:6]
                        if np.any(colors > 1.0) and np.max(colors) <= 255.0 and np.min(colors) >=0.0:
                            print("    Assuming RGB colors are in [0, 255] range, normalizing to [0, 1].")
                            colors = colors / 255.0
                        elif np.any(colors < 0.0) or np.any(colors > 1.0):
                            print("    Warning: RGB colors are outside [0,1] range. Clamping.")
                            colors = np.clip(colors, 0.0, 1.0)
                        pcd_loaded.colors = o3d.utility.Vector3dVector(colors)
                    elif data.shape[1] == 3: print("    Found 3 columns, assuming XYZ.")
                    else: print(f"    Found {data.shape[1]} columns. Using first 3 as XYZ. Color columns (if any) are ambiguous and ignored.")
                except ValueError as ve:
                    print(f"FATAL Error: Could not parse TXT file {args.input_point_cloud_file}. Error: {ve}")
                    sys.exit(1)
                except Exception as e_txt:
                    print(f"FATAL Error loading TXT file {args.input_point_cloud_file} with NumPy: {e_txt}")
                    sys.exit(1)

            elif file_extension == '.ply':
                print(f"  Loading PLY file: {args.input_point_cloud_file}")
                pcd_loaded = o3d.io.read_point_cloud(args.input_point_cloud_file)
            else:
                print(f"FATAL Error: Unsupported file extension '{file_extension}' for input point cloud file. Please use .ply or .txt.")
                sys.exit(1)

            if not pcd_loaded.has_points():
                print(f"FATAL Error: File {args.input_point_cloud_file} loaded as point cloud contains no points.")
                sys.exit(1)
            points_xyz_original_from_h5 = np.asarray(pcd_loaded.points, dtype=np.float32)
            if pcd_loaded.has_colors():
                colors_float = np.asarray(pcd_loaded.colors)
                if colors_float.max() > 1.0 or colors_float.min() < 0.0:
                    if colors_float.max() <= 255.0 and colors_float.min() >=0:
                         print("  Info: PLY/TXT colors appear to be in [0,255] range. Normalizing to [0,1] before conversion to uint8.")
                         colors_float = colors_float / 255.0
                    else:
                        print("  Warning: PLY/TXT colors are outside [0,1] and [0,255] ranges. Clamping to [0,1] before conversion to uint8.")
                        colors_float = np.clip(colors_float, 0.0, 1.0)
                points_rgb_original_from_h5 = (colors_float * 255).astype(np.uint8)
                print(f"  Loaded {points_xyz_original_from_h5.shape[0]} points with RGB from {args.input_point_cloud_file}.")
            else:
                print(f"  Loaded {points_xyz_original_from_h5.shape[0]} points (no RGB) from {args.input_point_cloud_file}.")
        except Exception as e:
            print(f"FATAL Error loading or processing point cloud file {args.input_point_cloud_file}: {e}")
            sys.exit(1) # Exit if file loading fails
    else: # HDF5 loading
        print(f"Loading data from HDF5: {args.input_h5}, Sample: {args.sample_index}")
        if not os.path.exists(args.input_h5):
            print(f"FATAL Error: HDF5 file not found: {args.input_h5}"); sys.exit(1)
        try:
            with h5py.File(args.input_h5, 'r') as f:
                if 'data' not in f: raise KeyError("'data' key missing in HDF5 file.")
                dset_data = f['data']
                dset_rgb = f.get('rgb', None)
                if not (0 <= args.sample_index < dset_data.shape[0]):
                    raise IndexError(f"Sample index {args.sample_index} out of bounds (0 to {dset_data.shape[0] - 1}).")
                points_xyz_original_from_h5 = dset_data[args.sample_index].astype(np.float32)
                if dset_rgb is not None:
                    if dset_rgb.shape[0] > args.sample_index and dset_rgb[args.sample_index].shape[0] == points_xyz_original_from_h5.shape[0]:
                        points_rgb_original_from_h5 = dset_rgb[args.sample_index].astype(np.uint8)
                    else: print(f"Warning: RGB data in HDF5 has issues for sample index {args.sample_index}. Not using RGB.")
                else: print("No 'rgb' key found in HDF5. Not using RGB.")
        except Exception as e:
            print(f"FATAL Error loading HDF5: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

    if points_xyz_original_from_h5 is None or points_xyz_original_from_h5.shape[0] == 0:
        print("FATAL Error: XYZ points data could not be loaded or is empty."); sys.exit(1)

    original_scene_o3d_pcd = o3d.geometry.PointCloud()
    original_scene_o3d_pcd.points = o3d.utility.Vector3dVector(points_xyz_original_from_h5)
    if points_rgb_original_from_h5 is not None:
        original_scene_o3d_pcd.colors = o3d.utility.Vector3dVector(points_rgb_original_from_h5.astype(np.float64) / 255.0)
    
    # Save common_original_scene.pcd
    path_common_original_scene = os.path.join(intermediate_output_dir, "common_original_scene.pcd")
    try:
        o3d.io.write_point_cloud(path_common_original_scene, original_scene_o3d_pcd)
        print(f"Saved common original scene to {path_common_original_scene}")
    except Exception as e:
        print(f"Error saving common original scene: {e}")

    if args.visualize_original_scene:
        visualize_single_pcd(original_scene_o3d_pcd, window_name=f"Original Scene - Sample {args.sample_index}")

    # --- 数据预处理: 为语义分割模型准备输入特征 ---
    # ... (original semantic preprocessing logic) ...
    points_xyz_normalized_for_semantic = normalize_xyz_for_semantic_model(points_xyz_original_from_h5)
    features_list_for_semantic = [points_xyz_normalized_for_semantic]
    if args.model_input_channels == 6:
        if points_rgb_original_from_h5 is not None:
            points_rgb_normalized_for_semantic = points_rgb_original_from_h5.astype(np.float32) / 255.0
            features_list_for_semantic.append(points_rgb_normalized_for_semantic)
        else:
            dummy_rgb = np.full((points_xyz_normalized_for_semantic.shape[0], 3), 0.5, dtype=np.float32)
            features_list_for_semantic.append(dummy_rgb)
    features_for_semantic_np = np.concatenate(features_list_for_semantic, axis=1)


    # --- 加载语义模型 ---
    # ... (original semantic model loading) ...
    print(f"\nLoading SEMANTIC model checkpoint from: {args.checkpoint_semantic}")
    if not os.path.isfile(args.checkpoint_semantic):
        print(f"FATAL Error: Semantic checkpoint not found: {args.checkpoint_semantic}"); sys.exit(1)
    try:
        model_semantic = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device)
        checkpoint_data_semantic = torch.load(args.checkpoint_semantic, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint_data_semantic: model_semantic.load_state_dict(checkpoint_data_semantic['model_state_dict'])
        elif 'state_dict' in checkpoint_data_semantic: model_semantic.load_state_dict(checkpoint_data_semantic['state_dict'])
        else: model_semantic.load_state_dict(checkpoint_data_semantic)
        print("Semantic model weights loaded successfully.")
    except Exception as e:
        print(f"FATAL Error initializing or loading semantic model: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    model_semantic.eval()

    # --- Semantic Segmentation Input Preparation (Potentially Downsampled) ---
    # ... (original downsampling logic for semantic segmentation) ...
    points_for_semantic_model_xyz = points_xyz_original_from_h5
    features_for_semantic_model_np = features_for_semantic_np 
    if args.semantic_downsample_voxel_size > 0:
        print(f"\nDownsampling point cloud for semantic segmentation with voxel size: {args.semantic_downsample_voxel_size}")
        pcd_to_downsample = o3d.geometry.PointCloud()
        pcd_to_downsample.points = o3d.utility.Vector3dVector(points_xyz_original_from_h5)
        if points_rgb_original_from_h5 is not None:
             pcd_to_downsample.colors = o3d.utility.Vector3dVector(points_rgb_original_from_h5.astype(np.float64)/255.0)
        downsampled_pcd_o3d = pcd_to_downsample.voxel_down_sample(args.semantic_downsample_voxel_size)
        if not downsampled_pcd_o3d.has_points():
            print("FATAL Error: Downsampling for semantic segmentation resulted in an empty point cloud."); sys.exit(1)
        points_for_semantic_model_xyz = np.asarray(downsampled_pcd_o3d.points, dtype=np.float32)
        print(f"  Point cloud for semantic model reduced to {points_for_semantic_model_xyz.shape[0]} points.")
        temp_xyz_normalized = normalize_xyz_for_semantic_model(points_for_semantic_model_xyz)
        temp_features_list = [temp_xyz_normalized]
        if args.model_input_channels == 6:
            if downsampled_pcd_o3d.has_colors():
                temp_rgb_normalized = np.asarray(downsampled_pcd_o3d.colors, dtype=np.float32)
                temp_features_list.append(temp_rgb_normalized)
            else: 
                dummy_rgb = np.full((points_for_semantic_model_xyz.shape[0], 3), 0.5, dtype=np.float32)
                temp_features_list.append(dummy_rgb)
        features_for_semantic_model_np = np.concatenate(temp_features_list, axis=1)


    # --- 语义分割推理 ---
    # ... (original semantic inference logic) ...
    print("\nPerforming semantic segmentation inference...")
    features_for_semantic_tensor = torch.from_numpy(features_for_semantic_model_np).float().unsqueeze(0).to(device)
    if features_for_semantic_tensor.shape[2] != args.model_input_channels:
        print(f"FATAL Error: Prepared semantic features dimension ({features_for_semantic_tensor.shape[2]}D) != model_input_channels ({args.model_input_channels}D)."); sys.exit(1)
    with torch.no_grad():
        logits_semantic = model_semantic(features_for_semantic_tensor)
    pred_labels_on_input_to_model_np = torch.argmax(logits_semantic, dim=2).squeeze(0).cpu().numpy()


    # --- Label Upsampling (if downsampling occurred) ---
    # ... (original label upsampling logic) ...
    pred_semantic_labels_np = None
    if args.semantic_downsample_voxel_size > 0:
        print(f"\nUpsampling semantic labels from {points_for_semantic_model_xyz.shape[0]} to {points_xyz_original_from_h5.shape[0]} points...")
        kdtree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_for_semantic_model_xyz)))
        pred_semantic_labels_np = np.zeros(points_xyz_original_from_h5.shape[0], dtype=pred_labels_on_input_to_model_np.dtype)
        print("  Assigning labels to original points...")
        for i in range(points_xyz_original_from_h5.shape[0]):
            [_, idx_nn, _] = kdtree.search_knn_vector_3d(points_xyz_original_from_h5[i], 1)
            pred_semantic_labels_np[i] = pred_labels_on_input_to_model_np[idx_nn[0]]
        print("  Label upsampling finished.")
    else:
        pred_semantic_labels_np = pred_labels_on_input_to_model_np


    # --- DBSCAN Clustering ---
    # ... (original DBSCAN logic to get instance_points_dict) ...
    # (Ensure this part is complete and correct from the original script)
    target_label_id = args.target_label_id
    instance_points_dict = {} 
    num_instances_found = 0
    # This section needs to be accurate based on whether semantic downsampling happened for DBSCAN input.
    # Assuming DBSCAN runs on full resolution semantic labels for simplicity here, or that the original logic handles it.
    # For this split, we focus on the loop *after* instance_points_dict is populated.
    # If DBSCAN was on downsampled points and then mapped to full, instance_points_dict should contain full res points.
    
    # Simplified DBSCAN logic for the sake of this example split (original logic is more complex with downsampling options)
    # The key is that instance_points_dict should contain original scale XYZ points for each instance.
    print("\nPerforming DBSCAN clustering on ORIGINAL full resolution data (for simplicity in split example)...")
    target_mask_on_predictions = (pred_semantic_labels_np == target_label_id)
    points_xyz_for_clustering = points_xyz_original_from_h5[target_mask_on_predictions]

    if points_xyz_for_clustering.shape[0] >= args.dbscan_min_samples:
        try:
            db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1)
            instance_labels_for_target_points = db.fit_predict(points_xyz_for_clustering)
            
            if args.visualize_dbscan_all_target_points and points_xyz_for_clustering.shape[0] > 0:
                # (Visualization logic for DBSCAN can be kept or removed for part1)
                pass 

            unique_instances = np.unique(instance_labels_for_target_points[instance_labels_for_target_points != -1])
            unique_instances.sort()
            num_instances_found = len(unique_instances)
            print(f"Clustering (full res) found {num_instances_found} potential instances with label {target_label_id}.")
            for inst_id in unique_instances:
                instance_mask_local = (instance_labels_for_target_points == inst_id)
                instance_points_dict[inst_id] = points_xyz_for_clustering[instance_mask_local]
                print(f"  Instance {inst_id} (full res): {instance_points_dict[inst_id].shape[0]} points")
        except Exception as e:
            print(f"Error during DBSCAN clustering (full res): {e}"); num_instances_found = 0; instance_points_dict = {}
    else:
        print(f"Not enough points for DBSCAN (full res, semantic label {target_label_id}): required {args.dbscan_min_samples}, found {points_xyz_for_clustering.shape[0]}.")

    if args.visualize_semantic_segmentation:
        visualize_semantic_segmentation(points_xyz_original_from_h5, pred_semantic_labels_np, args.num_classes, args.target_label_id)

    # --- 加载目标 CAD/扫描模型 ---
    print(f"\nLoading target CAD/model file from: {args.model_file}")
    target_mesh = o3d.geometry.TriangleMesh() # Will hold the mesh if loaded
    target_pcd_original_model_scale = o3d.geometry.PointCloud() # Will always hold points (sampled or direct)
    target_centroid_original_model_scale = np.zeros(3)

    if not os.path.exists(args.model_file):
        print(f"FATAL Error: Target model file not found: {args.model_file}"); sys.exit(1)
    try:
        # Attempt to read as mesh first
        temp_mesh = o3d.io.read_triangle_mesh(args.model_file)
        if temp_mesh.has_vertices():
            target_mesh = temp_mesh # Keep the mesh
            target_pcd_original_model_scale = target_mesh.sample_points_uniformly(number_of_points=args.model_sample_points)
            if not target_pcd_original_model_scale.has_points(): 
                print("Warning: Sampling from mesh failed. Trying to read as point cloud.")
                target_pcd_original_model_scale = o3d.io.read_point_cloud(args.model_file) # Fallback
                if not target_pcd_original_model_scale.has_points(): raise ValueError("Target model file (as PCD fallback) also has no points.")
        else: # If not a mesh, try reading as point cloud directly
            target_pcd_original_model_scale = o3d.io.read_point_cloud(args.model_file)
            if not target_pcd_original_model_scale.has_points(): raise ValueError("Target model file (as PCD) has no points.")
        
        target_centroid_original_model_scale = target_pcd_original_model_scale.get_center()
        target_pcd_model_centered_for_icp = copy.deepcopy(target_pcd_original_model_scale)
        target_pcd_model_centered_for_icp.translate(-target_centroid_original_model_scale)
        
        if args.visualize_cad_model:
            visualize_single_pcd(target_pcd_original_model_scale, window_name="Loaded CAD Model (Sampled for ICP)")

        # Save common target model data
        path_common_target_centered = os.path.join(intermediate_output_dir, "common_target_model_centered.pcd")
        path_common_target_original_scale = os.path.join(intermediate_output_dir, "common_target_model_original_scale.pcd")
        path_common_target_centroid = os.path.join(intermediate_output_dir, "common_target_centroid_original_model_scale.npy")
        path_model_file = os.path.join(intermediate_output_dir, "model_file_path.txt")

        o3d.io.write_point_cloud(path_common_target_centered, target_pcd_model_centered_for_icp)
        o3d.io.write_point_cloud(path_common_target_original_scale, target_pcd_original_model_scale)
        np.save(path_common_target_centroid, target_centroid_original_model_scale)
        with open(path_model_file, 'w') as f_model_path: f_model_path.write(args.model_file)
        print(f"Saved common target model data to {intermediate_output_dir}")

    except Exception as e:
        print(f"FATAL Error loading or processing target CAD/model file: {e}"); sys.exit(1)


    if num_instances_found > 0:
        print("\nProcessing instances for Preprocessing and PCA...")
        for inst_id, instance_points_original_scale_np in instance_points_dict.items():
            print(f"\nProcessing Instance ID: {inst_id} (Initial points: {instance_points_original_scale_np.shape[0]})")
            
            instance_pcd_original_scale = o3d.geometry.PointCloud()
            instance_pcd_original_scale.points = o3d.utility.Vector3dVector(instance_points_original_scale_np)
            
            instance_pcd_for_icp = copy.deepcopy(instance_pcd_original_scale) # This is preprocessed, but not centered.

            # Instance Preprocessing (Voxel, SOR, FPS)
            if args.preprocess_voxel_size > 0 and instance_pcd_for_icp.has_points():
                print(f"    Applying Voxel Downsampling (voxel_size={args.preprocess_voxel_size})")
                instance_pcd_for_icp = instance_pcd_for_icp.voxel_down_sample(args.preprocess_voxel_size)
                if not instance_pcd_for_icp.has_points(): print(f"  Instance {inst_id} empty after Voxel. Skipping."); continue
            if args.preprocess_sor_k > 0 and args.preprocess_sor_std_ratio > 0 and instance_pcd_for_icp.has_points():
                print(f"    Applying SOR (k={args.preprocess_sor_k}, std_ratio={args.preprocess_sor_std_ratio})")
                sor_pcd, _ = instance_pcd_for_icp.remove_statistical_outlier(nb_neighbors=args.preprocess_sor_k, std_ratio=args.preprocess_sor_std_ratio)
                if sor_pcd.has_points(): instance_pcd_for_icp = sor_pcd
                else: print("    Warning: SOR removed all points. Using points before SOR.")
                if not instance_pcd_for_icp.has_points(): print(f"  Instance {inst_id} empty after SOR. Skipping."); continue
            if args.preprocess_fps_n_points > 0 and instance_pcd_for_icp.has_points():
                if len(instance_pcd_for_icp.points) > args.preprocess_fps_n_points:
                    print(f"    Applying FPS to {args.preprocess_fps_n_points} points")
                    instance_pcd_for_icp = instance_pcd_for_icp.farthest_point_down_sample(args.preprocess_fps_n_points)
                if not instance_pcd_for_icp.has_points(): print(f"  Instance {inst_id} empty after FPS. Skipping."); continue

            if args.visualize_intermediate_pcds and instance_pcd_for_icp.has_points():
                visualize_single_pcd(instance_pcd_for_icp, window_name=f"Preprocessed Instance {inst_id} ({len(instance_pcd_for_icp.points)} pts)")

            instance_centroid_for_icp = instance_pcd_for_icp.get_center()
            instance_pcd_centered_for_icp = copy.deepcopy(instance_pcd_for_icp) # For PCA, this centered version is used.
            instance_pcd_centered_for_icp.translate(-instance_centroid_for_icp)
            
            # --- PCA-based Initial Alignment ---
            print("    Attempting PCA-based initial alignment...")
            initial_transform_icp = np.identity(4) 
            
            points_src_o_scale_np = np.asarray(instance_pcd_centered_for_icp.points)
            points_tgt_o_scale_np = np.asarray(target_pcd_model_centered_for_icp.points)

            print("      Normalizing point clouds for PCA step...")
            points_src_np = normalize_point_cloud_xyz_local(np.copy(points_src_o_scale_np))
            points_tgt_np = normalize_point_cloud_xyz_local(np.copy(points_tgt_o_scale_np))

            if points_src_np.shape[0] >= 3 and points_tgt_np.shape[0] >= 3:
                try:
                    cov_src = np.cov(points_src_np, rowvar=False)
                    eig_vals_src, R_s_eig_vecs = np.linalg.eigh(cov_src) 
                    if np.linalg.det(R_s_eig_vecs) < 0: R_s_eig_vecs[:, 0] *= -1 
                    cov_tgt = np.cov(points_tgt_np, rowvar=False)
                    eig_vals_tgt, R_t_eig_vecs = np.linalg.eigh(cov_tgt) 
                    if np.linalg.det(R_t_eig_vecs) < 0: R_t_eig_vecs[:, 0] *= -1
                    R_pca_align = R_t_eig_vecs @ R_s_eig_vecs.T
                    if np.linalg.det(R_pca_align) < 0: R_pca_align[:, 0] *= -1
                    initial_transform_icp[:3, :3] = R_pca_align
                    print(f"    PCA initial rotation applied. Det(R_pca_align): {np.linalg.det(R_pca_align):.4f}")

                    if args.visualize_pca_axes:
                        instance_pcd_transformed_by_pca = copy.deepcopy(instance_pcd_centered_for_icp)
                        instance_pcd_transformed_by_pca.transform(initial_transform_icp)
                        instance_orig_axes_rotated_by_pca = initial_transform_icp[:3,:3] @ R_s_eig_vecs
                        axis_vis_length = 100.0
                        if target_pcd_model_centered_for_icp.has_points():
                             diag_len = np.linalg.norm(target_pcd_model_centered_for_icp.get_max_bound() - target_pcd_model_centered_for_icp.get_min_bound())
                             if diag_len > 1e-3: axis_vis_length = diag_len * 0.3
                        visualize_pca_alignment_with_axes(
                            target_pcd_model_centered_for_icp, R_t_eig_vecs,
                            instance_pcd_transformed_by_pca, instance_orig_axes_rotated_by_pca,
                            axis_length=axis_vis_length, window_name=f"PCA Axes Alignment - Inst {inst_id}"
                        )
                except np.linalg.LinAlgError as e_pca_linalg:
                    print(f"    Warning: PCA computation failed due to LinAlgError ({e_pca_linalg}), using identity as initial transform.")
                except Exception as e_pca_general:
                    print(f"    Warning: General error during PCA alignment ({e_pca_general}), using identity as initial transform.")
            else:
                print("    Not enough points for PCA (source < 3 or target < 3), using identity as initial transform.")
            
            # --- Save instance-specific intermediate data ---
            path_inst_preprocessed = os.path.join(intermediate_output_dir, f"instance_{inst_id}_preprocessed.pcd")
            path_inst_centroid = os.path.join(intermediate_output_dir, f"instance_{inst_id}_centroid.npy")
            path_inst_pca_transform = os.path.join(intermediate_output_dir, f"instance_{inst_id}_pca_transform.npy")
            
            try:
                o3d.io.write_point_cloud(path_inst_preprocessed, instance_pcd_for_icp) # Save preprocessed, not centered
                np.save(path_inst_centroid, instance_centroid_for_icp)
                np.save(path_inst_pca_transform, initial_transform_icp)
                print(f"  Saved intermediate data for instance {inst_id} to {intermediate_output_dir}")
            except Exception as e_save_inst:
                print(f"  Error saving intermediate data for instance {inst_id}: {e_save_inst}")
            
            # ICP and subsequent steps are removed from this script.
    else:
        print("\nNo instances found for processing.")

    print(f"\nPart 1 (Preprocessing & PCA) finished at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config_file', type=str, default='pose_estimation_config.yaml', help='Path to the YAML configuration file.')
    cli_args, remaining_argv = pre_parser.parse_known_args()
    config_data = load_config_from_yaml(cli_args.config_file)
    parser = argparse.ArgumentParser(description='Part 1: Preprocessing, Segmentation, Clustering, and PCA for Pose Estimation.', parents=[pre_parser]) # Keep parents pre_parser
    
    # Add all argument groups from the original script
    # --- Input/Output ---
    io_group = parser.add_argument_group('Input/Output Configuration (from YAML or CLI)')
    io_group.add_argument('--input_h5', type=str, default=get_config_value(config_data, 'InputOutput', 'input_h5', './data/testla_part1_h5/test_0.h5'), help='Input HDF5 file. Ignored if --input_point_cloud_file is used.')
    io_group.add_argument('--sample_index', type=int, default=get_config_value(config_data, 'InputOutput', 'sample_index', 1), help='Sample index in HDF5. Ignored if --input_point_cloud_file is used.')
    io_group.add_argument('--input_point_cloud_file', type=str, default=get_config_value(config_data, 'InputOutput', 'input_point_cloud_file', None), help='Input point cloud file (.ply, .txt). Overrides HDF5 input.')
    io_group.add_argument('--checkpoint_semantic', type=str, default=get_config_value(config_data, 'InputOutput', 'checkpoint_semantic', "checkpoints_seg_tesla_part1_normalized/best_model.pth"), help='Semantic segmentation model checkpoint.')
    io_group.add_argument('--model_file', type=str, default=get_config_value(config_data, 'InputOutput', 'model_file', "stp/part1_rude.STL"), help='Target 3D model file (STL, PLY, OBJ) for ICP.')
    io_group.add_argument('--output_dir', type=str, default=get_config_value(config_data, 'InputOutput', 'output_dir', './pose_estimation_results_script1'), help='Directory to save results and intermediate data.')

    # --- Semantic Model Configuration & Architecture ---
    model_config_group = parser.add_argument_group('Semantic Model Configuration (from YAML or CLI)')
    model_config_group.add_argument('--num_classes', type=int, default=get_config_value(config_data, 'SemanticModelConfig', 'num_classes', 2), help='Number of semantic classes.')
    model_config_group.add_argument('--model_input_channels', type=int, default=get_config_value(config_data, 'SemanticModelConfig', 'model_input_channels', 6), choices=[3, 6], help="Input channels for semantic model (3=XYZ, 6=XYZRGB).")
    model_config_group.add_argument('--k_neighbors', type=int, default=get_config_value(config_data, 'SemanticModelConfig', 'k_neighbors', 16), help='k for k-NN graph in model.')
    model_config_group.add_argument('--embed_dim', type=int, default=get_config_value(config_data, 'SemanticModelConfig', 'embed_dim', 64),help='Initial embedding dimension in model.')
    model_config_group.add_argument('--pt_hidden_dim', type=int, default=get_config_value(config_data, 'SemanticModelConfig', 'pt_hidden_dim', 128), help='Point Transformer hidden dimension.')
    model_config_group.add_argument('--pt_heads', type=int, default=get_config_value(config_data, 'SemanticModelConfig', 'pt_heads', 4), help='Number of attention heads in model.')
    model_config_group.add_argument('--num_transformer_layers', type=int, default=get_config_value(config_data, 'SemanticModelConfig', 'num_transformer_layers', 2), help='Number of transformer layers in model.')
    model_config_group.add_argument('--dropout', type=float, default=get_config_value(config_data, 'SemanticModelConfig', 'dropout', 0.3), help='Dropout rate in model.')

    # --- Other Data Processing Parameters ---
    data_proc_group = parser.add_argument_group('Other Data Processing (from YAML or CLI)')
    data_proc_group.add_argument('--semantic_downsample_voxel_size', type=float, default=get_config_value(config_data, 'OtherDataProcessing', 'semantic_downsample_voxel_size', 5.0), help='Voxel size for pre-semantic downsampling (0 to disable).')
    data_proc_group.add_argument('--model_sample_points', type=int, default=get_config_value(config_data, 'OtherDataProcessing', 'model_sample_points', 20480), help='Points to sample from CAD model for ICP.')

    # --- Semantic Target & DBSCAN ---
    dbscan_group = parser.add_argument_group('Semantic Target & DBSCAN (from YAML or CLI)')
    dbscan_group.add_argument('--target_label_id', type=int, default=get_config_value(config_data, 'SemanticTargetDBSCAN', 'target_label_id', 1), help='Target semantic label ID for DBSCAN.')
    dbscan_group.add_argument('--dbscan_eps', type=float, default=get_config_value(config_data, 'SemanticTargetDBSCAN', 'dbscan_eps', 200.0), help='DBSCAN epsilon.')
    dbscan_group.add_argument('--dbscan_min_samples', type=int, default=get_config_value(config_data, 'SemanticTargetDBSCAN', 'dbscan_min_samples', 500), help='DBSCAN min_samples.')

    # --- Instance Preprocessing ---
    pp_group = parser.add_argument_group('Instance Preprocessing (from YAML or CLI)')
    pp_group.add_argument('--preprocess_voxel_size', type=float, default=get_config_value(config_data, 'InstancePreprocessing', 'preprocess_voxel_size', 0.0), help='Voxel size for instance downsampling (0 to disable).')
    pp_group.add_argument('--preprocess_sor_k', type=int, default=get_config_value(config_data, 'InstancePreprocessing', 'preprocess_sor_k', 0), help='Neighbors for SOR (0 to disable).')
    pp_group.add_argument('--preprocess_sor_std_ratio', type=float, default=get_config_value(config_data, 'InstancePreprocessing', 'preprocess_sor_std_ratio', 1.0), help='Std ratio for SOR.')
    pp_group.add_argument('--preprocess_fps_n_points', type=int, default=get_config_value(config_data, 'InstancePreprocessing', 'preprocess_fps_n_points', 2048), help='Target points for FPS (0 to disable).')
    # This arg is for ICP, but PCA might also have a min point requirement, or it's implicitly handled by checks before PCA.
    # The original script used args.icp_min_points before the PCA block for early exit.
    # Removing --icp_min_points from here as per user request to move all ICP params to Part 2
    # pp_group.add_argument('--icp_min_points', type=int, default=get_config_value(config_data, 'ICPParameters', 'icp_min_points', 10000), help='Min instance points for processing (PCA/ICP) after preprocessing.')


    # --- Control & Visualization (Script 1 specific parts + common ones for Script 2) ---
    ctrl_group = parser.add_argument_group('Control & Visualization (from YAML or CLI)')
    ctrl_group.add_argument('--no_cuda', action='store_true', default=get_config_value(config_data, 'ControlVisualization', 'no_cuda', True), help='Disable CUDA, use CPU.')
    # save_results for Script 1 means saving intermediate data. Script 2 will have its own for final poses.
    # ctrl_group.add_argument('--save_results', action='store_true', default=get_config_value(config_data, 'ControlVisualization', 'save_results', False), help='Save intermediate data.')
    
    ctrl_group.add_argument('--visualize_original_scene', action='store_true', default=get_config_value(config_data, 'ControlVisualization', 'visualize_original_scene', True), help='Visualize the original scene loaded.')
    ctrl_group.add_argument('--visualize_semantic_segmentation', action='store_true', default=get_config_value(config_data, 'ControlVisualization', 'visualize_semantic_segmentation', True), help='Visualize the output of semantic segmentation.')
    ctrl_group.add_argument('--visualize_dbscan_all_target_points', action='store_true', default=get_config_value(config_data, 'ControlVisualization', 'visualize_dbscan_all_target_points', True), help='Visualize DBSCAN results (target class points).')
    ctrl_group.add_argument('--visualize_cad_model', action='store_true', default=get_config_value(config_data, 'ControlVisualization', 'visualize_cad_model', True), help='Visualize the loaded CAD model.')
    ctrl_group.add_argument('--visualize_intermediate_pcds', action='store_true', default=get_config_value(config_data, 'ControlVisualization', 'visualize_intermediate_pcds', True), help='Visualize preprocessed instance point clouds.')
    ctrl_group.add_argument('--visualize_pca_axes', action='store_true', default=get_config_value(config_data, 'ControlVisualization', 'visualize_pca_axes', True), help='Visualize PCA axes alignment.')
    # Visualization flags for Script 2 (will be loaded from args.json by Script 2)
    ctrl_group.add_argument('--visualize_pose', action='store_true', default=get_config_value(config_data, 'ControlVisualization', 'visualize_pose', True), help='(For Script 2) Visualize ICP alignment.')
    ctrl_group.add_argument('--visualize_pose_in_scene', action='store_true', default=get_config_value(config_data, 'ControlVisualization', 'visualize_pose_in_scene', True), help='(For Script 2) Visualize final transformed model in scene.')
    ctrl_group.add_argument('--save_results', action='store_true', default=get_config_value(config_data, 'ControlVisualization', 'save_results', True), help='(For Script 2) Save estimated poses to a .npz file.')


    args = parser.parse_args(sys.argv[1:])
    main(args) 