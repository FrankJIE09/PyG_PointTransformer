# _estimate_pose_from_h5_step6.py
# 版本: 统一语义分割模型加载逻辑，使其与 _evaluate_step5.py 类似。
# 对输入给语义模型的XYZ坐标进行归一化。
# 修改: 默认激活所有可视化过程。

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
    if not scene_vis.has_colors(): scene_vis.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
    if isinstance(model_vis, o3d.geometry.TriangleMesh):
        model_vis.paint_uniform_color([0, 1, 0])  # Green
        if not model_vis.has_vertex_normals(): model_vis.compute_vertex_normals()
    elif isinstance(model_vis, o3d.geometry.PointCloud):
        model_vis.paint_uniform_color([0, 1, 0])  # Green

    print(f"\nDisplaying: {window_name}")
    print("Scene Color (e.g., Gray/Original RGB): Original Full Scene | Green: Transformed Target Model")
    print("(Close the Open3D window to continue script execution...)")

    geometries_to_draw = []
    if scene_vis.has_points():  # Only add scene if it has points
        geometries_to_draw.append(scene_vis)
    else:
        print(f"Warning: Original scene for '{window_name}' has no points.")

    if (isinstance(model_vis, o3d.geometry.TriangleMesh) and model_vis.has_vertices()) or \
            (isinstance(model_vis, o3d.geometry.PointCloud) and model_vis.has_points()):
        geometries_to_draw.append(model_vis)
    else:
        print(f"Warning: Transformed model for '{window_name}' has no geometry, not drawing it.")

    if len(geometries_to_draw) > 0:  # Check if there's anything to draw
        try:
            o3d.visualization.draw_geometries(geometries_to_draw, window_name=window_name)
            print(f"Visualization window '{window_name}' closed.")
        except Exception as e:
            print(f"Error during visualization of '{window_name}': {e}")
    else:
        print(f"Not enough valid geometries to display for '{window_name}'.")


def visualize_semantic_segmentation(points_xyz, semantic_labels_np, num_classes, target_label_id,
                                    window_name="Semantic Segmentation Output"):
    """ 可视化语义分割结果 """
    if not OPEN3D_AVAILABLE: print("Open3D not available, skipping visualization."); return
    if points_xyz.shape[0] == 0: print(f"Skipping visualization for {window_name}: No points."); return

    print(f"\nDisplaying Semantic Segmentation: {window_name}")
    print("(Close the Open3D window to continue script execution...)")

    semantic_pcd = o3d.geometry.PointCloud()
    semantic_pcd.points = o3d.utility.Vector3dVector(points_xyz)

    # 为不同语义标签生成颜色
    # 你可以自定义一个颜色映射表
    max_label = np.max(semantic_labels_np) if semantic_labels_np.size > 0 else -1

    # 使用预定义的 distinct_colors 列表 (如果 _evaluate_step5.py 中有，或者在此处定义)
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
            colors_for_labels[i] = [1, 0, 0]  # 目标类别用鲜红色
        else:
            colors_for_labels[i] = distinct_colors[i % num_distinct_colors]  # 其他类别循环使用颜色
            if i >= num_distinct_colors:  # 如果预定义颜色不够，用随机颜色补充（但尽量避免）
                colors_for_labels[i] = np.random.rand(3)

    if semantic_labels_np.size > 0:
        # Handle cases where semantic_labels_np might contain labels > max_label (should not happen with argmax)
        # or if max_label is -1 (empty semantic_labels_np, though we check points_xyz.shape[0] earlier)
        valid_labels_mask = semantic_labels_np <= max_label
        point_colors = np.zeros((semantic_labels_np.shape[0], 3))
        if np.any(valid_labels_mask):  # only apply colors if there are valid labels
            point_colors[valid_labels_mask] = colors_for_labels[semantic_labels_np[valid_labels_mask]]
        semantic_pcd.colors = o3d.utility.Vector3dVector(point_colors)
    else:
        semantic_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Default if no labels

    try:
        o3d.visualization.draw_geometries([semantic_pcd], window_name=window_name)
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
        # To differentiate slightly if they overlap perfectly, we use the same RGB colors.
        # If alignment is perfect, instance axes lines will superimpose on target axes lines.
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
    print(f"Starting Pose Estimation from HDF5 at: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Arguments: {args}")

    if not OPEN3D_AVAILABLE:  # Should have exited earlier, but double check
        print("Open3D is essential for this script. Exiting.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)

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
                print(f"  Loading TXT file: {args.input_point_cloud_file} using NumPy...")
                try:
                    # Try to detect delimiter: space or comma
                    with open(args.input_point_cloud_file, 'r') as f_test_delim:
                        first_line = f_test_delim.readline()
                    delimiter = ',' if ',' in first_line else None # Default to space/whitespace if no comma

                    data = np.loadtxt(args.input_point_cloud_file, delimiter=delimiter, dtype=np.float32)
                    if data.ndim == 1: # handle case of single point
                        data = data.reshape(1, -1)
                    
                    if data.shape[0] == 0: # Check if file is empty after loading
                        raise ValueError("TXT file is empty or could not be parsed.")
                    if data.shape[1] < 3:
                        raise ValueError(f"TXT file must have at least 3 columns for XYZ. Found {data.shape[1]} columns.")
                    
                    pcd_loaded = o3d.geometry.PointCloud()
                    pcd_loaded.points = o3d.utility.Vector3dVector(data[:, :3])
                    
                    if data.shape[1] >= 6: # Potential RGB
                        print(f"    Found {data.shape[1]} columns. Assuming first 3 are XYZ, next 3 are RGB.")
                        colors = data[:, 3:6]
                        # Normalize colors if they are in 0-255 range
                        if np.any(colors > 1.0) and np.max(colors) <= 255.0 and np.min(colors) >=0.0:
                            print("    Assuming RGB colors are in [0, 255] range, normalizing to [0, 1].")
                            colors = colors / 255.0
                        elif np.any(colors < 0.0) or np.any(colors > 1.0): # If not 0-255, clamp to 0-1 if they are outside
                            print("    Warning: RGB colors are outside [0,1] range. Clamping.")
                            colors = np.clip(colors, 0.0, 1.0)
                        pcd_loaded.colors = o3d.utility.Vector3dVector(colors)
                    elif data.shape[1] == 3:
                        print("    Found 3 columns, assuming XYZ.")
                    else: # 4 or 5 columns, ambiguous
                        print(f"    Found {data.shape[1]} columns. Using first 3 as XYZ. Color columns (if any) are ambiguous and ignored.")
                                
                except ValueError as ve: # Handles wrong number of columns from loadtxt or our checks
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
                # Ensure colors are in [0,1] range before multiplying by 255
                if colors_float.max() > 1.0 or colors_float.min() < 0.0:
                    # Check if they might already be 0-255
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
                # points_rgb_original_from_h5 remains None

        except Exception as e:
            print(f"FATAL Error loading or processing point cloud file {args.input_point_cloud_file}: {e}")
    else:
        print(f"Loading data from HDF5: {args.input_h5}, Sample: {args.sample_index}")
        if not os.path.exists(args.input_h5):
            print(f"FATAL Error: HDF5 file not found: {args.input_h5}");
            sys.exit(1)
        try:
            with h5py.File(args.input_h5, 'r') as f:
                if 'data' not in f: raise KeyError("'data' key missing in HDF5 file.")
                dset_data = f['data']
                dset_rgb = f.get('rgb', None)
                if not (0 <= args.sample_index < dset_data.shape[0]):
                    raise IndexError(f"Sample index {args.sample_index} out of bounds (0 to {dset_data.shape[0] - 1}).")
                points_xyz_original_from_h5 = dset_data[args.sample_index].astype(np.float32)
                if dset_rgb is not None:
                    if dset_rgb.shape[0] > args.sample_index and \
                            dset_rgb[args.sample_index].shape[0] == points_xyz_original_from_h5.shape[0]:
                        points_rgb_original_from_h5 = dset_rgb[args.sample_index].astype(np.uint8)
                    else:
                        print(
                            f"Warning: RGB data in HDF5 has issues for sample index {args.sample_index}. Not using RGB for this sample.")
                else:
                    print("No 'rgb' key found in HDF5. Not using RGB for this sample.")
        except Exception as e:
            print(f"FATAL Error loading HDF5: {e}");
            import traceback;
            traceback.print_exc();
            sys.exit(1)

    if points_xyz_original_from_h5 is None or points_xyz_original_from_h5.shape[0] == 0:
        print("FATAL Error: XYZ points data could not be loaded or is empty.");
        sys.exit(1)

    # --- 创建原始场景点云对象 (供后续可视化使用) ---
    original_scene_o3d_pcd = o3d.geometry.PointCloud()
    original_scene_o3d_pcd.points = o3d.utility.Vector3dVector(points_xyz_original_from_h5)
    if points_rgb_original_from_h5 is not None:
        original_scene_o3d_pcd.colors = o3d.utility.Vector3dVector(
            points_rgb_original_from_h5.astype(np.float64) / 255.0
        )
    else: # 如果没有RGB，赋予一个默认颜色，避免可视化函数内部的额外检查和默认着色逻辑的潜在不一致
        pass # visualize_single_pcd 和 visualize_transformed_model_in_scene 会处理无色的情况

    # 可视化原始加载的场景点云 (如果参数启用)
    if args.visualize_original_scene:
        visualize_single_pcd(original_scene_o3d_pcd, window_name=f"Original Scene - Sample {args.sample_index}")

    # --- 数据预处理: 为语义分割模型准备输入特征 ---
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
    print(f"\nLoading SEMANTIC model checkpoint from: {args.checkpoint_semantic}")
    if not os.path.isfile(args.checkpoint_semantic):
        print(f"FATAL Error: Semantic checkpoint not found: {args.checkpoint_semantic}");
        sys.exit(1)
    try:
        model_semantic = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device)
        checkpoint_data_semantic = torch.load(args.checkpoint_semantic, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint_data_semantic:
            model_semantic.load_state_dict(checkpoint_data_semantic['model_state_dict'])
        elif 'state_dict' in checkpoint_data_semantic:
            model_semantic.load_state_dict(checkpoint_data_semantic['state_dict'])
        else:
            model_semantic.load_state_dict(checkpoint_data_semantic)
        print("Semantic model weights loaded successfully.")
    except Exception as e:
        print(f"FATAL Error initializing or loading semantic model: {e}");
        import traceback;
        traceback.print_exc();
        sys.exit(1)
    model_semantic.eval()

    # --- Semantic Segmentation Input Preparation (Potentially Downsampled) ---
    points_for_semantic_model_xyz = points_xyz_original_from_h5
    features_for_semantic_model_np = features_for_semantic_np # This was already prepared based on original_from_h5
    original_indices_for_downsampled = None # To map labels back if downsampling occurs
    full_pcd_o3d_for_kdtree = None # For kdtree if upsampling labels

    if args.semantic_downsample_voxel_size > 0:
        print(f"\nDownsampling point cloud for semantic segmentation with voxel size: {args.semantic_downsample_voxel_size}")
        full_pcd_o3d_for_kdtree = o3d.geometry.PointCloud() # Create once for kdtree
        full_pcd_o3d_for_kdtree.points = o3d.utility.Vector3dVector(points_xyz_original_from_h5)
        if points_rgb_original_from_h5 is not None:
            # We need to ensure RGB is also downsampled consistently if present
            # For simplicity in this pass, we re-create features for the downsampled cloud
            # This assumes normalize_xyz_for_semantic_model and feature prep can run on fewer points
            pcd_to_downsample = o3d.geometry.PointCloud()
            pcd_to_downsample.points = o3d.utility.Vector3dVector(points_xyz_original_from_h5)
            if points_rgb_original_from_h5 is not None:
                 pcd_to_downsample.colors = o3d.utility.Vector3dVector(points_rgb_original_from_h5.astype(np.float64)/255.0)
            
            # Perform voxel downsampling and retain original indices if possible (Open3D >= 0.10.0)
            # For simplicity, we'll just downsample points and features, then upsample labels via kdtree
            downsampled_pcd_o3d = pcd_to_downsample.voxel_down_sample(args.semantic_downsample_voxel_size)
            
            if not downsampled_pcd_o3d.has_points():
                print("FATAL Error: Downsampling for semantic segmentation resulted in an empty point cloud. Try a smaller voxel size.")
                sys.exit(1)
            
            points_for_semantic_model_xyz = np.asarray(downsampled_pcd_o3d.points, dtype=np.float32)
            print(f"  Point cloud for semantic model reduced to {points_for_semantic_model_xyz.shape[0]} points.")

            # Re-prepare features for the downsampled point cloud
            temp_xyz_normalized = normalize_xyz_for_semantic_model(points_for_semantic_model_xyz)
            temp_features_list = [temp_xyz_normalized]
            if args.model_input_channels == 6:
                if downsampled_pcd_o3d.has_colors():
                    temp_rgb_normalized = np.asarray(downsampled_pcd_o3d.colors, dtype=np.float32) # Already [0,1]
                    temp_features_list.append(temp_rgb_normalized)
                else: # Should not happen if original had RGB and we re-created, but as a fallback
                    dummy_rgb = np.full((points_for_semantic_model_xyz.shape[0], 3), 0.5, dtype=np.float32)
                    temp_features_list.append(dummy_rgb)
            features_for_semantic_model_np = np.concatenate(temp_features_list, axis=1)
        else: # No RGB, just downsample XYZ
            points_for_semantic_model_xyz = np.asarray(full_pcd_o3d_for_kdtree.voxel_down_sample(args.semantic_downsample_voxel_size).points, dtype=np.float32)
            if points_for_semantic_model_xyz.shape[0] == 0:
                print("FATAL Error: Downsampling for semantic segmentation resulted in an empty point cloud. Try a smaller voxel size.")
                sys.exit(1)
            print(f"  Point cloud for semantic model reduced to {points_for_semantic_model_xyz.shape[0]} points (XYZ only).")
            # Re-prepare features (XYZ only)
            features_for_semantic_model_np = normalize_xyz_for_semantic_model(points_for_semantic_model_xyz)
    
    # --- 语义分割推理 --- (On potentially downsampled data)
    print("\nPerforming semantic segmentation inference...")
    features_for_semantic_tensor = torch.from_numpy(features_for_semantic_model_np).float().unsqueeze(0).to(device)
    if features_for_semantic_tensor.shape[2] != args.model_input_channels:
        print(
            f"FATAL Error: Prepared semantic features dimension ({features_for_semantic_tensor.shape[2]}D) != model_input_channels ({args.model_input_channels}D).");
        sys.exit(1)
    with torch.no_grad():
        logits_semantic = model_semantic(features_for_semantic_tensor)
    pred_labels_on_input_to_model_np = torch.argmax(logits_semantic, dim=2).squeeze(0).cpu().numpy()

    # --- Label Upsampling (if downsampling occurred) ---
    pred_semantic_labels_np = None # Final labels for the original full point cloud
    pred_instance_labels_full_np = None # Final instance labels for the original full point cloud (after potential upsampling)

    # --- DBSCAN Clustering (Potentially on Downsampled Data) ---
    target_label_id = args.target_label_id
    instance_points_dict = {} # This will store final, full-resolution instance points
    num_instances_found = 0

    if args.semantic_downsample_voxel_size > 0: # DBSCAN on downsampled, then map instances to full cloud
        print("\nPerforming DBSCAN clustering on DOWN SAMPLED data...")
        # points_for_semantic_model_xyz are the downsampled points
        # pred_labels_on_input_to_model_np are their semantic labels
        target_mask_downsampled = (pred_labels_on_input_to_model_np == target_label_id)
        points_xyz_for_dbscan_downsampled = points_for_semantic_model_xyz[target_mask_downsampled]
        
        # We need to know the original indices (in points_for_semantic_model_xyz) of points_xyz_for_dbscan_downsampled
        # to correctly assign DBSCAN labels back to the (downsampled) semantic points that were clustered.
        original_indices_of_downsampled_target_points = np.where(target_mask_downsampled)[0]

        dbscan_instance_labels_for_downsampled_target_points = np.full(points_for_semantic_model_xyz.shape[0], -1, dtype=int) # Init all as noise

        if points_xyz_for_dbscan_downsampled.shape[0] >= args.dbscan_min_samples:
            try:
                db_downsampled = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1)
                # These labels are only for points_xyz_for_dbscan_downsampled
                temp_dbscan_labels = db_downsampled.fit_predict(points_xyz_for_dbscan_downsampled)
                
                # Assign these temp_dbscan_labels back to the correct positions in dbscan_instance_labels_for_downsampled_target_points
                dbscan_instance_labels_for_downsampled_target_points[original_indices_of_downsampled_target_points] = temp_dbscan_labels
                
                # Visualize DBSCAN on downsampled data if requested
                if args.visualize_dbscan_all_target_points and points_xyz_for_dbscan_downsampled.shape[0] > 0:
                    dbscan_vis_pcd_downsampled = o3d.geometry.PointCloud()
                    dbscan_vis_pcd_downsampled.points = o3d.utility.Vector3dVector(points_xyz_for_dbscan_downsampled)
                    # Color by temp_dbscan_labels
                    max_inst_lbl_ds = np.max(temp_dbscan_labels) if temp_dbscan_labels.size > 0 else -1
                    colors_dbscan_ds = np.random.rand(max_inst_lbl_ds + 2, 3)
                    colors_dbscan_ds[0] = [0.5,0.5,0.5]
                    # Use distinct_colors from global scope or re-define if needed
                    distinct_colors_local = [
                        [230/255, 25/255, 75/255], [60/255, 180/255, 75/255], [255/255, 225/255, 25/255],
                        [0/255, 130/255, 200/255], [245/255, 130/255, 48/255], [145/255, 30/255, 180/255]
                    ]
                    for lbl_idx_ds in range(max_inst_lbl_ds + 1):
                        colors_dbscan_ds[lbl_idx_ds + 1] = distinct_colors_local[lbl_idx_ds % len(distinct_colors_local)]
                    dbscan_vis_pcd_downsampled.colors = o3d.utility.Vector3dVector(colors_dbscan_ds[temp_dbscan_labels + 1])
                    visualize_single_pcd(dbscan_vis_pcd_downsampled, window_name=f"DBSCAN on Downsampled Target {target_label_id}")

            except Exception as e_db_downsampled:
                print(f"Error during DBSCAN on downsampled data: {e_db_downsampled}")
        else:
            print(f"Not enough points for DBSCAN on downsampled data (semantic label {target_label_id}): required {args.dbscan_min_samples}, found {points_xyz_for_dbscan_downsampled.shape[0]}.")

        # Now, upsample both semantic labels and these new DBSCAN instance labels
        print(f"\nUpsampling semantic AND DBSCAN instance labels from {points_for_semantic_model_xyz.shape[0]} points to {points_xyz_original_from_h5.shape[0]} points...")
        kdtree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_for_semantic_model_xyz)))
        pred_semantic_labels_np = np.zeros(points_xyz_original_from_h5.shape[0], dtype=pred_labels_on_input_to_model_np.dtype)
        pred_instance_labels_full_np = np.full(points_xyz_original_from_h5.shape[0], -1, dtype=int) # Init all as noise

        print("  Assigning labels to original points (this may take a moment for large clouds)...")
        for i in range(points_xyz_original_from_h5.shape[0]):
            [_, idx_nn, _] = kdtree.search_knn_vector_3d(points_xyz_original_from_h5[i], 1)
            nearest_neighbor_original_index_in_downsampled = idx_nn[0]
            pred_semantic_labels_np[i] = pred_labels_on_input_to_model_np[nearest_neighbor_original_index_in_downsampled]
            # Only assign instance label if the semantic label also matches (important guard)
            if pred_semantic_labels_np[i] == target_label_id:
                 pred_instance_labels_full_np[i] = dbscan_instance_labels_for_downsampled_target_points[nearest_neighbor_original_index_in_downsampled]
        print("  Label and Instance ID upsampling finished.")
        
        # Extract full-resolution instances based on pred_instance_labels_full_np
        unique_instances_full = np.unique(pred_instance_labels_full_np[pred_instance_labels_full_np != -1])
        num_instances_found = len(unique_instances_full)
        print(f"Clustering (on downsampled, mapped to full) found {num_instances_found} potential instances with label {target_label_id}.")
        for inst_id_full in unique_instances_full:
            instance_mask_full = (pred_instance_labels_full_np == inst_id_full) & (pred_semantic_labels_np == target_label_id)
            # Ensure we are using original full resolution points here
            current_instance_points_full_res = points_xyz_original_from_h5[instance_mask_full]
            if current_instance_points_full_res.shape[0] > 0:
                 instance_points_dict[inst_id_full] = current_instance_points_full_res
                 print(f"  Instance {inst_id_full} (full res): {current_instance_points_full_res.shape[0]} points")
            else:
                print(f"  Warning: Instance {inst_id_full} has 0 points after mapping to full resolution. Skipping.")
                if inst_id_full in unique_instances_full: num_instances_found -=1 # Decrement if it becomes empty

    else: # Original behavior: DBSCAN on full resolution data
        print("\nPerforming DBSCAN clustering on ORIGINAL full resolution data...")
        pred_semantic_labels_np = pred_labels_on_input_to_model_np # Predictions were already on full cloud
        target_mask_on_predictions = (pred_semantic_labels_np == target_label_id)
        points_xyz_for_clustering = points_xyz_original_from_h5[target_mask_on_predictions]

        if points_xyz_for_clustering.shape[0] >= args.dbscan_min_samples:
            try:
                db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1)
                instance_labels_for_target_points = db.fit_predict(points_xyz_for_clustering)
                
                # Visualization of DBSCAN on full data (existing logic)
                if args.visualize_dbscan_all_target_points and points_xyz_for_clustering.shape[0] > 0:
                    dbscan_vis_pcd = o3d.geometry.PointCloud()
                    dbscan_vis_pcd.points = o3d.utility.Vector3dVector(points_xyz_for_clustering)
                    max_instance_label = np.max(instance_labels_for_target_points) if instance_labels_for_target_points.size > 0 else -1
                    colors_dbscan = np.random.rand(max_instance_label + 2, 3)
                    colors_dbscan[0] = [0.5, 0.5, 0.5]
                    distinct_colors_local = [
                        [230/255, 25/255, 75/255], [60/255, 180/255, 75/255], [255/255, 225/255, 25/255],
                        [0/255, 130/255, 200/255], [245/255, 130/255, 48/255], [145/255, 30/255, 180/255]
                    ]
                    for lbl_idx in range(max_instance_label + 1):
                        colors_dbscan[lbl_idx + 1] = distinct_colors_local[lbl_idx % len(distinct_colors_local)]
                    dbscan_vis_pcd.colors = o3d.utility.Vector3dVector(colors_dbscan[instance_labels_for_target_points + 1])
                    visualize_single_pcd(dbscan_vis_pcd, window_name=f"DBSCAN Output (Label {target_label_id} points - Full Res)")

                unique_instances = np.unique(instance_labels_for_target_points[instance_labels_for_target_points != -1])
                unique_instances.sort()
                num_instances_found = len(unique_instances)
                print(f"Clustering (full res) found {num_instances_found} potential instances with label {target_label_id}.")
                for inst_id in unique_instances:
                    instance_mask_local = (instance_labels_for_target_points == inst_id)
                    instance_points_dict[inst_id] = points_xyz_for_clustering[instance_mask_local]
                    print(f"  Instance {inst_id} (full res): {instance_points_dict[inst_id].shape[0]} points")
            except Exception as e:
                print(f"Error during DBSCAN clustering (full res): {e}");
                num_instances_found = 0;
                instance_points_dict = {}
        else:
            print(f"Not enough points for DBSCAN (full res, semantic label {target_label_id}): required {args.dbscan_min_samples}, found {points_xyz_for_clustering.shape[0]}.")

    unique_semantic_full, counts_semantic_full = np.unique(pred_semantic_labels_np, return_counts=True)
    print(f"  Predicted semantic label distribution (on full cloud, after any upsampling): {dict(zip(unique_semantic_full, counts_semantic_full))}")

    # 可视化语义分割结果 (如果参数启用)
    if args.visualize_semantic_segmentation:
        # 使用原始（未归一化）的XYZ坐标进行可视化，以便观察真实尺度下的分割效果
        visualize_semantic_segmentation(points_xyz_original_from_h5, pred_semantic_labels_np, args.num_classes,
                                        args.target_label_id)

    # --- 加载目标 CAD/扫描模型 ---
    print(f"\nLoading target CAD/model file from: {args.model_file}")
    target_mesh = o3d.geometry.TriangleMesh()
    target_pcd_original_model_scale = o3d.geometry.PointCloud()
    target_centroid_original_model_scale = np.zeros(3)
    if not os.path.exists(args.model_file):
        print(f"FATAL Error: Target model file not found: {args.model_file}");
        sys.exit(1)
    try:
        temp_mesh = o3d.io.read_triangle_mesh(args.model_file)
        if temp_mesh.has_vertices():
            target_mesh = temp_mesh
            target_pcd_original_model_scale = target_mesh.sample_points_uniformly(
                number_of_points=args.model_sample_points)
            if not target_pcd_original_model_scale.has_points(): raise ValueError("Sampling from mesh failed.")
        else:
            target_pcd_original_model_scale = o3d.io.read_point_cloud(args.model_file)
            if not target_pcd_original_model_scale.has_points(): raise ValueError(
                "Target model file (as PCD) has no points.")
        target_centroid_original_model_scale = target_pcd_original_model_scale.get_center()
        target_pcd_model_centered_for_icp = copy.deepcopy(target_pcd_original_model_scale)
        target_pcd_model_centered_for_icp.translate(-target_centroid_original_model_scale)
        if args.visualize_cad_model:  # Visualize the loaded (and sampled if mesh) CAD model
            visualize_single_pcd(target_pcd_original_model_scale, window_name="Loaded CAD Model (Sampled for ICP)")
    except Exception as e:
        print(f"FATAL Error loading or processing target CAD/model file: {e}");
        sys.exit(1)

    # --- Configure ICP Parameters ---
    print("\nConfiguring ICP parameters...")
    threshold_icp = args.icp_threshold
    method_str_icp = args.icp_estimation_method

    estimated_poses = {}
    if num_instances_found > 0:
        print("\nPerforming ICP for each found instance...")
        for inst_id, instance_points_original_scale_np in instance_points_dict.items():
            print(f"\nProcessing Instance ID: {inst_id} (Initial points: {instance_points_original_scale_np.shape[0]})")
            instance_pcd_original_scale = o3d.geometry.PointCloud()
            instance_pcd_original_scale.points = o3d.utility.Vector3dVector(instance_points_original_scale_np)
            instance_pcd_for_icp = copy.deepcopy(instance_pcd_original_scale)

            # Instance Preprocessing
            if args.preprocess_voxel_size > 0 and instance_pcd_for_icp.has_points():
                print(f"    Applying Voxel Downsampling (voxel_size={args.preprocess_voxel_size})")
                instance_pcd_for_icp = instance_pcd_for_icp.voxel_down_sample(args.preprocess_voxel_size)
                if not instance_pcd_for_icp.has_points(): print(
                    f"  Instance {inst_id} empty after Voxel. Skipping."); continue
            if args.preprocess_sor_k > 0 and args.preprocess_sor_std_ratio > 0 and instance_pcd_for_icp.has_points():
                print(f"    Applying SOR (k={args.preprocess_sor_k}, std_ratio={args.preprocess_sor_std_ratio})")
                sor_pcd, _ = instance_pcd_for_icp.remove_statistical_outlier(nb_neighbors=args.preprocess_sor_k,
                                                                             std_ratio=args.preprocess_sor_std_ratio)
                if sor_pcd.has_points():
                    instance_pcd_for_icp = sor_pcd
                else:
                    print("    Warning: SOR removed all points. Using points before SOR.")
                if not instance_pcd_for_icp.has_points(): print(
                    f"  Instance {inst_id} empty after SOR. Skipping."); continue
            if args.preprocess_fps_n_points > 0 and instance_pcd_for_icp.has_points():
                if len(instance_pcd_for_icp.points) > args.preprocess_fps_n_points:
                    print(f"    Applying FPS to {args.preprocess_fps_n_points} points")
                    instance_pcd_for_icp = instance_pcd_for_icp.farthest_point_down_sample(args.preprocess_fps_n_points)
                if not instance_pcd_for_icp.has_points(): print(
                    f"  Instance {inst_id} empty after FPS. Skipping."); continue

            if args.visualize_intermediate_pcds and instance_pcd_for_icp.has_points():
                visualize_single_pcd(instance_pcd_for_icp,
                                     window_name=f"Preprocessed Instance {inst_id} for ICP ({len(instance_pcd_for_icp.points)} pts)")

            if len(instance_pcd_for_icp.points) < args.icp_min_points:
                print(
                    f"  Skipping ICP for Inst {inst_id}: Not enough points after preprocessing ({len(instance_pcd_for_icp.points)} < {args.icp_min_points}).");
                continue

            instance_centroid_for_icp = instance_pcd_for_icp.get_center()
            instance_pcd_centered_for_icp = copy.deepcopy(instance_pcd_for_icp)
            instance_pcd_centered_for_icp.translate(-instance_centroid_for_icp)

            estimation_method_icp = o3d_reg.TransformationEstimationPointToPoint()
            if method_str_icp.lower() == 'point_to_plane':
                normal_radius_icp = threshold_icp * 2.0;
                normal_radius_icp = max(1e-3, normal_radius_icp)
                try:
                    print(f"    Estimating normals for instance (radius={normal_radius_icp:.4f})")
                    instance_pcd_centered_for_icp.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_icp, max_nn=30))
                    if not target_pcd_model_centered_for_icp.has_normals():
                        print(f"    Estimating normals for target model (radius={normal_radius_icp:.4f})")
                        target_pcd_model_centered_for_icp.estimate_normals(
                            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_icp, max_nn=30))
                    
                    if instance_pcd_centered_for_icp.has_normals() and target_pcd_model_centered_for_icp.has_normals():
                        print("    Using PointToPlane ICP.")
                        estimation_method_icp = o3d_reg.TransformationEstimationPointToPlane()
                    else:
                        print("    Warning: Failed to estimate normals or one pcd lacks normals. Falling back to PointToPoint ICP.")
                        estimation_method_icp = o3d_reg.TransformationEstimationPointToPoint() # Fallback
                except Exception as e_normal:
                    print(f"    Warning: Error estimating normals ({e_normal}). Falling back to PointToPoint ICP.")
                    estimation_method_icp = o3d_reg.TransformationEstimationPointToPoint() # Fallback
            else:
                print("    Using PointToPoint ICP as per configuration.")

            criteria_icp = o3d_reg.ICPConvergenceCriteria(relative_fitness=args.icp_relative_fitness,
                                                          relative_rmse=args.icp_relative_rmse,
                                                          max_iteration=args.icp_max_iter)
            
            # --- PCA-based Initial Alignment ---
            print("    Attempting PCA-based initial alignment...")
            initial_transform_icp = np.identity(4) # Default if PCA fails or not enough points
            
            points_src_o_scale_np = np.asarray(instance_pcd_centered_for_icp.points) # Original scale, centered
            points_tgt_o_scale_np = np.asarray(target_pcd_model_centered_for_icp.points) # Original scale, centered

            # Normalize for PCA to make it scale-invariant for rotation determination
            print("      Normalizing point clouds for PCA step...")
            points_src_np = normalize_point_cloud_xyz_local(np.copy(points_src_o_scale_np)) # Use local normalizer
            points_tgt_np = normalize_point_cloud_xyz_local(np.copy(points_tgt_o_scale_np)) # Use local normalizer

            if points_src_np.shape[0] >= 3 and points_tgt_np.shape[0] >= 3:
                try:
                    # Source PCA
                    cov_src = np.cov(points_src_np, rowvar=False)
                    eig_vals_src, R_s_eig_vecs = np.linalg.eigh(cov_src) 
                    if np.linalg.det(R_s_eig_vecs) < 0:
                        R_s_eig_vecs[:, 0] *= -1 

                    # Target PCA
                    cov_tgt = np.cov(points_tgt_np, rowvar=False)
                    eig_vals_tgt, R_t_eig_vecs = np.linalg.eigh(cov_tgt) 
                    if np.linalg.det(R_t_eig_vecs) < 0:
                        R_t_eig_vecs[:, 0] *= -1
                    
                    # Rotation to align source's principal axes with target's principal axes
                    # R_pca_align transforms source such that its new axes are aligned with target's axes
                    R_pca_align = R_t_eig_vecs @ R_s_eig_vecs.T
                    
                    # Ensure the final rotation is proper (should be if R_t_eig_vecs and R_s_eig_vecs are proper)
                    if np.linalg.det(R_pca_align) < 0:
                         print("      Warning: Determinant of PCA rotation matrix is negative after composition. This is unexpected. Flipping one axis of R_pca_align.")
                         R_pca_align[:, 0] *= -1 # Attempt to fix, though ideally this shouldn't be needed.

                    initial_transform_icp[:3, :3] = R_pca_align
                    print(f"    PCA initial rotation applied. Det(R_pca_align): {np.linalg.det(R_pca_align):.4f}")
                    print(f"    PCA Rotation Matrix (R_t @ R_s.T):{R_pca_align}")

                    if args.visualize_pca_axes:
                        # pcd_target_centered is target_pcd_model_centered_for_icp
                        # target_axes_vectors are columns of R_t_eig_vecs
                        
                        # pcd_instance_pca_aligned needs to be created
                        instance_pcd_transformed_by_pca = copy.deepcopy(instance_pcd_centered_for_icp)
                        instance_pcd_transformed_by_pca.transform(initial_transform_icp) # initial_transform_icp has R_pca_align

                        # instance_transformed_axes_vectors are columns of R_pca_align @ R_s_eig_vecs
                        # R_s_eig_vecs are columns: R_s_eig_vecs
                        # R_pca_align is initial_transform_icp[:3,:3]
                        instance_orig_axes_rotated_by_pca = initial_transform_icp[:3,:3] @ R_s_eig_vecs
                        
                        axis_vis_length = 100.0 # Default length in mm, adjust as needed
                        # Try to make axis length somewhat dynamic, e.g., 30% of max extent of target
                        if target_pcd_model_centered_for_icp.has_points():
                             diag_len = np.linalg.norm(target_pcd_model_centered_for_icp.get_max_bound() - target_pcd_model_centered_for_icp.get_min_bound())
                             if diag_len > 1e-3: # Check if diag_len is valid
                                axis_vis_length = diag_len * 0.3


                        visualize_pca_alignment_with_axes(
                            target_pcd_model_centered_for_icp, R_t_eig_vecs,
                            instance_pcd_transformed_by_pca, instance_orig_axes_rotated_by_pca,
                            axis_length=axis_vis_length, # Adjust this value based on your point cloud scale
                            window_name=f"PCA Axes Alignment - Inst {inst_id}"
                        )

                except np.linalg.LinAlgError as e_pca_linalg:
                    print(f"    Warning: PCA computation failed due to LinAlgError ({e_pca_linalg}), using identity as initial transform.")
                    initial_transform_icp = np.identity(4)
                except Exception as e_pca_general:
                    print(f"    Warning: General error during PCA alignment ({e_pca_general}), using identity as initial transform.")
                    initial_transform_icp = np.identity(4)
            else:
                print("    Not enough points for PCA (source < 3 or target < 3), using identity as initial transform.")
                initial_transform_icp = np.identity(4)
            # --- End PCA-based Initial Alignment ---

            try:
                # Ensure threshold_icp is available here, it's args.icp_threshold 
                current_icp_threshold = args.icp_threshold 

                print(f"    Running ICP with threshold: {current_icp_threshold:.4f}, max_iter: {args.icp_max_iter}")
                start_time_icp = time.perf_counter() # Record start time
                
                reg_result = o3d_reg.registration_icp(instance_pcd_centered_for_icp, target_pcd_model_centered_for_icp,
                                                      current_icp_threshold, initial_transform_icp, estimation_method_icp,
                                                      criteria_icp)
                
                end_time_icp = time.perf_counter() # Record end time
                duration_icp = end_time_icp - start_time_icp # Calculate duration

                T_centered_s_to_centered_t = reg_result.transformation
                T_world_to_instance_centroid = np.eye(4);
                T_world_to_instance_centroid[:3, 3] = instance_centroid_for_icp
                T_model_centroid_to_world = np.eye(4); T_model_centroid_to_world[:3, 3] = -target_centroid_original_model_scale
                final_estimated_pose = T_world_to_instance_centroid @ T_centered_s_to_centered_t @ T_model_centroid_to_world
                estimated_poses[inst_id] = final_estimated_pose
                print(f"  ICP for Inst {inst_id}:")
                print(f"    ICP Duration    : {duration_icp:.4f} seconds") # Print ICP duration
                print(f"    Fitness         : {reg_result.fitness:.6f}")
                print(f"    Inlier RMSE     : {reg_result.inlier_rmse:.6f}")
                print(f"    Correspondence Set: {len(reg_result.correspondence_set)} pairs")
                print(f"    Transformation (centered_instance to centered_model):\n{reg_result.transformation}")

                if args.visualize_pose:
                    transformed_cad_pcd_for_viz = copy.deepcopy(target_pcd_original_model_scale);
                    transformed_cad_pcd_for_viz.transform(final_estimated_pose)
                    visualize_icp_alignment(transformed_cad_pcd_for_viz, instance_pcd_for_icp,
                                            window_name=f"ICP Align - Inst {inst_id}")
                if args.visualize_pose_in_scene:
                    model_geometry_for_scene_viz = target_mesh if target_mesh.has_vertices() else target_pcd_original_model_scale
                    visualize_transformed_model_in_scene(original_scene_o3d_pcd, model_geometry_for_scene_viz,
                                                         final_estimated_pose,
                                                         window_name=f"Final CAD in Scene - Inst {inst_id}")
            except Exception as e_icp_run:
                print(f"Error during ICP run for instance {inst_id}: {e_icp_run}")
    else:
        print("\nNo instances for ICP.")

    if args.save_results and estimated_poses:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        pose_filename = os.path.join(args.output_dir, f"estimated_poses_h5sample_{args.sample_index}_{timestamp}.npz")
        try:
            np.savez(pose_filename, **{f'instance_{k}': v for k, v in estimated_poses.items()}); print(
                f"\nSaved poses to {pose_filename}")
        except Exception as e_save:
            print(f"\nError saving poses: {e_save}")
    print("\nPose estimation script finished.")


if __name__ == "__main__":
    # Initial parser to get --config_file argument first
    # This allows the config file path to be specified on the command line
    pre_parser = argparse.ArgumentParser(add_help=False) # Disable help to avoid conflict with main parser
    pre_parser.add_argument('--config_file', type=str, default='pose_estimation_config.yaml',
                            help='Path to the YAML configuration file.')
    cli_args, remaining_argv = pre_parser.parse_known_args() # Parse only known args (i.e., --config_file)

    # Load config from YAML file specified by --config_file or its default
    config_data = load_config_from_yaml(cli_args.config_file)

    # Main parser that will use YAML values as defaults
    parser = argparse.ArgumentParser(
        description='Perform Instance Segmentation and ICP Pose Estimation from various inputs.',
        # parents=[pre_parser] # We add --config_file manually to the main parser as well for help string
    )
    
    # --- Input/Output ---
    io_group = parser.add_argument_group('Input/Output Configuration (from YAML or CLI)')
    io_group.add_argument('--config_file', type=str, 
                          default=cli_args.config_file, # Default is what pre_parser found or its default
                          help='Path to the YAML configuration file. CLI overrides YAML.')
    io_group.add_argument('--input_h5', type=str, 
                          default=get_config_value(config_data, 'InputOutput', 'input_h5', './data/testla_part1_h5/test_0.h5'), 
                          help='Input HDF5 file. Ignored if --input_point_cloud_file is used.')
    io_group.add_argument('--sample_index', type=int, 
                          default=get_config_value(config_data, 'InputOutput', 'sample_index', 0), 
                          help='Sample index in HDF5. Ignored if --input_point_cloud_file is used.')
    io_group.add_argument('--input_point_cloud_file', type=str, 
                          default=get_config_value(config_data, 'InputOutput', 'input_point_cloud_file', './data/point_cloud_00002.txt'),
                          help='Input point cloud file (.ply, .txt). Overrides HDF5 input.')
    io_group.add_argument('--checkpoint_semantic', type=str,
                          default=get_config_value(config_data, 'InputOutput', 'checkpoint_semantic', "checkpoints_seg_tesla_part1_normalized/best_model.pth"),
                          help='Semantic segmentation model checkpoint.')
    io_group.add_argument('--model_file', type=str, 
                          default=get_config_value(config_data, 'InputOutput', 'model_file', "stp/part1_rude.STL"),
                          help='Target 3D model file (STL, PLY, OBJ) for ICP.')
    io_group.add_argument('--output_dir', type=str, 
                          default=get_config_value(config_data, 'InputOutput', 'output_dir', './pose_estimation_results_cmdline'),
                          help='Directory to save results.')

    # --- Semantic Model Configuration & Architecture ---
    model_config_group = parser.add_argument_group('Semantic Model Configuration (from YAML or CLI)')
    model_config_group.add_argument('--num_classes', type=int, 
                                    default=get_config_value(config_data, 'SemanticModelConfig', 'num_classes', 2),
                                    help='Number of semantic classes.')
    model_config_group.add_argument('--model_input_channels', type=int, 
                                    default=get_config_value(config_data, 'SemanticModelConfig', 'model_input_channels', 6), choices=[3, 6],
                                    help="Input channels for semantic model (3=XYZ, 6=XYZRGB).")
    model_config_group.add_argument('--k_neighbors', type=int, 
                                    default=get_config_value(config_data, 'SemanticModelConfig', 'k_neighbors', 16), help='k for k-NN graph in model.')
    model_config_group.add_argument('--embed_dim', type=int, 
                                    default=get_config_value(config_data, 'SemanticModelConfig', 'embed_dim', 64),
                                    help='Initial embedding dimension in model.')
    model_config_group.add_argument('--pt_hidden_dim', type=int, 
                                    default=get_config_value(config_data, 'SemanticModelConfig', 'pt_hidden_dim', 128),
                                    help='Point Transformer hidden dimension.')
    model_config_group.add_argument('--pt_heads', type=int, 
                                    default=get_config_value(config_data, 'SemanticModelConfig', 'pt_heads', 4), help='Number of attention heads in model.')
    model_config_group.add_argument('--num_transformer_layers', type=int, 
                                    default=get_config_value(config_data, 'SemanticModelConfig', 'num_transformer_layers', 2),
                                    help='Number of transformer layers in model.')
    model_config_group.add_argument('--dropout', type=float, 
                                    default=get_config_value(config_data, 'SemanticModelConfig', 'dropout', 0.3),
                                    help='Dropout rate in model.')

    # --- Other Data Processing Parameters ---
    data_proc_group = parser.add_argument_group('Other Data Processing (from YAML or CLI)')
    data_proc_group.add_argument('--semantic_downsample_voxel_size', type=float, 
                                 default=get_config_value(config_data, 'OtherDataProcessing', 'semantic_downsample_voxel_size', 5.0),
                                 help='Voxel size for pre-semantic downsampling (0 to disable).')
    data_proc_group.add_argument('--model_sample_points', type=int, 
                                 default=get_config_value(config_data, 'OtherDataProcessing', 'model_sample_points', 20480),
                                 help='Points to sample from CAD model for ICP.')

    # --- Semantic Target & DBSCAN ---
    dbscan_group = parser.add_argument_group('Semantic Target & DBSCAN (from YAML or CLI)')
    dbscan_group.add_argument('--target_label_id', type=int, 
                              default=get_config_value(config_data, 'SemanticTargetDBSCAN', 'target_label_id', 1), 
                              help='Target semantic label ID for DBSCAN.')
    dbscan_group.add_argument('--dbscan_eps', type=float, 
                              default=get_config_value(config_data, 'SemanticTargetDBSCAN', 'dbscan_eps', 200.0),
                              help='DBSCAN epsilon.')
    dbscan_group.add_argument('--dbscan_min_samples', type=int, 
                              default=get_config_value(config_data, 'SemanticTargetDBSCAN', 'dbscan_min_samples', 500), 
                              help='DBSCAN min_samples.')

    # --- Instance Preprocessing ---
    pp_group = parser.add_argument_group('Instance Preprocessing (from YAML or CLI)')
    pp_group.add_argument('--preprocess_voxel_size', type=float, 
                          default=get_config_value(config_data, 'InstancePreprocessing', 'preprocess_voxel_size', 0.0),
                          help='Voxel size for instance downsampling (0 to disable).')
    pp_group.add_argument('--preprocess_sor_k', type=int, 
                          default=get_config_value(config_data, 'InstancePreprocessing', 'preprocess_sor_k', 0), 
                          help='Neighbors for SOR (0 to disable).')
    pp_group.add_argument('--preprocess_sor_std_ratio', type=float, 
                          default=get_config_value(config_data, 'InstancePreprocessing', 'preprocess_sor_std_ratio', 1.0),
                          help='Std ratio for SOR.')
    pp_group.add_argument('--preprocess_fps_n_points', type=int, 
                          default=get_config_value(config_data, 'InstancePreprocessing', 'preprocess_fps_n_points', 2048),
                          help='Target points for FPS (0 to disable).')

    # --- ICP Parameters ---
    icp_group = parser.add_argument_group('ICP Parameters (from YAML or CLI)')
    icp_group.add_argument('--icp_threshold', type=float, 
                           default=get_config_value(config_data, 'ICPParameters', 'icp_threshold', 25.0), 
                           help='ICP max_correspondence_distance.')
    icp_group.add_argument('--icp_estimation_method', type=str, 
                           default=get_config_value(config_data, 'ICPParameters', 'icp_estimation_method', 'point_to_plane'),
                           choices=['point_to_point', 'point_to_plane'], help="ICP estimation method.")
    icp_group.add_argument('--icp_relative_fitness', type=float, 
                           default=get_config_value(config_data, 'ICPParameters', 'icp_relative_fitness', 1e-8),
                           help='ICP convergence: relative fitness.')
    icp_group.add_argument('--icp_relative_rmse', type=float, 
                           default=get_config_value(config_data, 'ICPParameters', 'icp_relative_rmse', 1e-8), 
                           help='ICP convergence: relative RMSE.')
    icp_group.add_argument('--icp_max_iter', type=int, 
                           default=get_config_value(config_data, 'ICPParameters', 'icp_max_iter', 2000), 
                           help='ICP convergence: max iterations.')
    icp_group.add_argument('--icp_min_points', type=int, 
                           default=get_config_value(config_data, 'ICPParameters', 'icp_min_points', 100),
                           help='Min instance points for ICP after preprocessing.')

    # --- Control & Visualization ---
    ctrl_group = parser.add_argument_group('Control & Visualization (from YAML or CLI)')
    # For boolean flags that are True by default via config (like most visualize flags):
    # We use store_true, and their default from YAML will be True. If user specifies --flag on CLI, it remains True.
    # If user wants to set them False, they must do so in YAML or we need to add --no-visualize_x flags.
    # For flags like `no_cuda` or `save_results` which are False by default in YAML:
    # `default=False` from YAML for `action='store_true'` means they are False unless flag is on CLI.
    ctrl_group.add_argument('--no_cuda', action='store_true', 
                            default=get_config_value(config_data, 'ControlVisualization', 'no_cuda', False), 
                            help='Disable CUDA, use CPU.')
    ctrl_group.add_argument('--save_results', action='store_true', 
                            default=get_config_value(config_data, 'ControlVisualization', 'save_results', False), 
                            help='Save estimated poses to a .npz file.')
    
    # Visualization flags: default to True as per YAML. User can't turn them OFF from CLI with this setup easily.
    # To allow turning them off from CLI, one would typically have pairs: --visualize_x and --no-visualize_x
    # Or, make them type=bool and parse 'true'/'false' strings. For simplicity, keeping store_true with YAML default.
    ctrl_group.add_argument('--visualize_original_scene', action='store_true', 
                            default=get_config_value(config_data, 'ControlVisualization', 'visualize_original_scene', True),
                            help='Visualize the original scene loaded.')
    ctrl_group.add_argument('--visualize_semantic_segmentation', action='store_true', 
                            default=get_config_value(config_data, 'ControlVisualization', 'visualize_semantic_segmentation', True),
                            help='Visualize the output of semantic segmentation.')
    ctrl_group.add_argument('--visualize_dbscan_all_target_points', action='store_true', 
                            default=get_config_value(config_data, 'ControlVisualization', 'visualize_dbscan_all_target_points', True),
                            help='Visualize DBSCAN results (target class points).' )
    ctrl_group.add_argument('--visualize_cad_model', action='store_true', 
                            default=get_config_value(config_data, 'ControlVisualization', 'visualize_cad_model', True),
                            help='Visualize the loaded CAD model.')
    ctrl_group.add_argument('--visualize_intermediate_pcds', action='store_true', 
                            default=get_config_value(config_data, 'ControlVisualization', 'visualize_intermediate_pcds', True),
                            help='Visualize preprocessed instance point clouds.')
    ctrl_group.add_argument('--visualize_pose', action='store_true', 
                            default=get_config_value(config_data, 'ControlVisualization', 'visualize_pose', True),
                            help='Visualize ICP alignment.')
    ctrl_group.add_argument('--visualize_pose_in_scene', action='store_true', 
                            default=get_config_value(config_data, 'ControlVisualization', 'visualize_pose_in_scene', True),
                            help='Visualize final transformed model in scene.')
    ctrl_group.add_argument('--visualize_pca_axes', action='store_true', 
                            default=get_config_value(config_data, 'ControlVisualization', 'visualize_pca_axes', True),
                            help='Visualize PCA axes alignment.')

    # Parse all arguments. CLI arguments will override defaults (which are now from YAML or hardcoded fallbacks).
    args = parser.parse_args(sys.argv[1:]) # Use sys.argv[1:] to avoid script name being parsed as an arg.
    # If --config_file was in remaining_argv from pre_parser, it will be handled correctly here.

    main(args)