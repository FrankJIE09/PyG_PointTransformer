# segment_instances_interactive_hardcoded.py
# 从 HDF5 加载 (硬编码路径和索引)，执行语义分割，
# 然后使用 Open3D GUI 通过滑块交互式调整 DBSCAN 参数进行聚类。

import torch
import numpy as np
import argparse # Still needed for remaining args
import time
import os
import random
import datetime
import sys
import h5py

# --- 导入本地模块 ---
try:
    # Make sure model.py is in the Python path or current directory
    from model import PyG_PointTransformerSegModel # 语义分割模型类
except ImportError as e:
    print(f"FATAL Error importing model (model.py): {e}")
    print("Ensure model.py is in the current directory or Python path.")
    sys.exit(1)

# --- 导入 Open3D 和 Scikit-learn ---
try:
    import open3d as o3d
    # Import specific GUI and rendering modules
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    OPEN3D_AVAILABLE = True
except ImportError:
    print("FATAL Error: Open3D >= 0.13 (with GUI support) is required for interactive mode.")
    OPEN3D_AVAILABLE = False
    sys.exit(1) # Exit if Open3D GUI is not available

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError: print("FATAL Error: scikit-learn not found."); sys.exit(1)


# --- 硬编码的输入/输出参数 ---
# Values previously set by argparse defaults
HARDCODED_INPUT_H5 = './data/my_custom_dataset_h5_rgb/test_0.h5'
HARDCODED_SAMPLE_INDEX = 0
HARDCODED_CHECKPOINT = 'checkpoints_seg_pyg_ptconv_rgb/best_model.pth'
HARDCODED_OUTPUT_DIR = './instance_results_interactive' # Using the interactive script's default

# --- 全局常量 ---
POINT_CLOUD_NAME = "point_cloud_geometry"
MARKER_BASE_NAME = "centroid_marker_"

# ==============================================================================
# --- 交互式应用类 ---
# ==============================================================================
class InstanceSegmenterApp:

    def __init__(self, args, points_xyz_original, points_processed, semantic_labels, input_h5_path, sample_idx):
        """
        Initializes the interactive segmentation application window.

        Args:
            args: Parsed command line arguments (for remaining parameters).
            points_xyz_original (np.ndarray): Original XYZ coordinates (N, 3), used for saving.
            points_processed (np.ndarray): Processed XYZ coordinates (N, 3) used for DBSCAN/Viz.
            semantic_labels (np.ndarray): Predicted semantic labels (N,).
            input_h5_path (str): Path to the HDF5 file being processed (for window title).
            sample_idx (int): Index of the sample being processed (for window title).
        """
        self.args = args # Store args for parameters like dbscan_eps, marker_radius etc.
        self.points_xyz_original = points_xyz_original
        self.points_processed = points_processed
        self.semantic_labels = semantic_labels
        self.instance_labels = np.full_like(self.semantic_labels, -1, dtype=np.int64)
        self.centroids = {}
        self.unique_instances = []
        self.current_eps = args.dbscan_eps # Initial value from args
        self.current_min_samples = args.dbscan_min_samples # Initial value from args
        self.current_marker_names = set() # Keep track of added markers

        # --- 初始化应用和窗口 ---
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window(
            f"Interactive Segmentation - {os.path.basename(input_h5_path)} Sample {sample_idx}",
            1280, 800
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close) # Handle saving on close

        # --- 3D 场景设置 ---
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([0.1, 0.1, 0.1, 1.0]) # Dark background
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(self.points_processed)
        )
        self.widget3d.setup_camera(60.0, bbox, bbox.get_center())
        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultLit" # Or "unlit" for no lighting influence
        self.material.point_size = 3 # Adjust point size if needed

        # --- 创建点云几何体 (只创建一次) ---
        self.pcd_geom = o3d.geometry.PointCloud()
        self.pcd_geom.points = o3d.utility.Vector3dVector(self.points_processed)
        # Initial coloring will happen in _update_clustering_and_viz
        self.widget3d.scene.add_geometry(POINT_CLOUD_NAME, self.pcd_geom, self.material)

        # --- GUI 控件 ---
        em = self.window.theme.font_size # For spacing/sizing

        # --- 设置面板 ---
        self.settings_panel = gui.Vert(0, gui.Margins(em // 2, em // 2, em // 2, em // 2))

        # --- DBSCAN Eps Slider ---
        self.settings_panel.add_child(gui.Label("DBSCAN Epsilon (eps)"))
        self.eps_slider = gui.Slider(gui.Slider.DOUBLE)
        # Determine reasonable range based on data, maybe needs adjustment
        min_eps = 0.001
        max_eps = max(min_eps + 0.1, self.current_eps * 5) # Heuristic max range based on initial value
        self.eps_slider.set_limits(min_eps, max_eps)
        self.eps_slider.double_value = self.current_eps
        self.eps_slider.set_on_value_changed(self._on_slider_change)
        self.settings_panel.add_child(self.eps_slider)
        self.eps_label = gui.Label(f"Value: {self.current_eps:.4f}")
        self.settings_panel.add_child(self.eps_label)

        # --- DBSCAN Min Samples Slider ---
        self.settings_panel.add_child(gui.Label("DBSCAN Min Samples"))
        self.min_samples_slider = gui.Slider(gui.Slider.INT)
        min_samp = 2
        max_samp = max(20, self.current_min_samples * 5) # Heuristic max range based on initial value
        self.min_samples_slider.set_limits(min_samp, max_samp)
        self.min_samples_slider.int_value = self.current_min_samples
        self.min_samples_slider.set_on_value_changed(self._on_slider_change)
        self.settings_panel.add_child(self.min_samples_slider)
        self.min_samples_label = gui.Label(f"Value: {self.current_min_samples}")
        self.settings_panel.add_child(self.min_samples_label)

        # --- Info Label ---
        self.info_label = gui.Label("Instance Info:")
        self.settings_panel.add_child(gui.VGrid(1,em //2)) # Spacer
        self.settings_panel.add_child(self.info_label)


        # --- Window Layout ---
        # Add settings panel and 3D widget to the window
        self.window.add_child(self.widget3d)
        self.window.add_child(self.settings_panel)

        # --- Perform Initial Clustering and Visualization ---
        self._update_clustering_and_viz()


    def _on_layout(self, layout_context):
        # Position widgets correctly on layout changes (e.g., window resize)
        r = self.window.content_rect
        width = 17 * layout_context.theme.font_size # Width for settings panel
        self.widget3d.frame = gui.Rect(r.x, r.y, r.get_right() - width, r.height)
        self.settings_panel.frame = gui.Rect(self.widget3d.frame.get_right(), r.y, width, r.height)

    def _on_slider_change(self, new_value):
        # Callback when either slider's value changes
        self.current_eps = self.eps_slider.double_value
        self.current_min_samples = self.min_samples_slider.int_value

        # Update labels displaying slider values
        self.eps_label.text = f"Value: {self.current_eps:.4f}"
        self.min_samples_label.text = f"Value: {self.current_min_samples}"

        # Re-run clustering and update the visualization
        # Use post_to_main_thread for safety, though likely not strictly needed here
        # as slider callbacks run on the main thread anyway.
        self.app.post_to_main_thread(self.window, self._update_clustering_and_viz)


    def _update_clustering_and_viz(self):
        """Performs DBSCAN clustering and updates the 3D visualization."""
        # print(f"\nUpdating clustering with eps={self.current_eps:.4f}, min_samples={self.current_min_samples}") # Reduce console noise

        # Filter points based on semantic prediction
        screw_label_id = self.args.screw_label_id
        screw_mask = (self.semantic_labels == screw_label_id)
        screw_points_xyz = self.points_processed[screw_mask]

        # Reset instance labels and centroids for the update
        self.instance_labels.fill(-1)
        self.centroids.clear()
        num_instances_found = 0
        num_noise_points = np.sum(screw_mask) # Initially, all screw points are potential noise
        self.unique_instances = [] # Reset unique instances list

        if screw_points_xyz.shape[0] >= self.current_min_samples:
            try:
                db = DBSCAN(eps=self.current_eps, min_samples=self.current_min_samples, n_jobs=-1)
                instance_labels_screw_points = db.fit_predict(screw_points_xyz)

                # Update the full instance label array
                self.instance_labels[screw_mask] = instance_labels_screw_points

                # Find unique instances (excluding noise label -1)
                self.unique_instances, counts = np.unique(instance_labels_screw_points[instance_labels_screw_points != -1], return_counts=True)
                num_instances_found = len(self.unique_instances)
                num_noise_points = np.sum(instance_labels_screw_points == -1)

                # print(f"  Clustering found {num_instances_found} instances and {num_noise_points} noise points (predicted as screw).") # Reduce console noise

                # Calculate centroids for found instances
                for i, inst_id in enumerate(self.unique_instances):
                    instance_mask_screw = (instance_labels_screw_points == inst_id)
                    instance_points = screw_points_xyz[instance_mask_screw]
                    if instance_points.shape[0] > 0:
                        centroid = np.mean(instance_points, axis=0)
                        self.centroids[inst_id] = centroid

            except Exception as e:
                print(f"Error during DBSCAN: {e}")
                # Keep instance_labels as -1, centroids empty, num_instances_found = 0
                num_instances_found = 0
                num_noise_points = np.sum(screw_mask) # Reset noise count potentially
                self.unique_instances = []

        else:
            # print(f"  Not enough points ({screw_points_xyz.shape[0]}) predicted as screw for DBSCAN with min_samples={self.current_min_samples}.") # Reduce noise
            # State is already reset (instance_labels=-1, centroids empty, etc.)
            num_instances_found = 0
            num_noise_points = np.sum(screw_mask)
            self.unique_instances = []


        # --- Update Visualization ---

        # 1. Update Point Cloud Colors
        max_instance_id = np.max(self.instance_labels) if num_instances_found > 0 else -1
        num_colors_needed = max_instance_id + 2
        num_colors_needed = max(num_colors_needed, len(self.centroids) + 1)

        rng_colors = np.random.default_rng(42)
        instance_colors = rng_colors.random((num_colors_needed, 3))
        instance_colors[0] = [0.5, 0.5, 0.5] # Gray for noise

        point_colors = instance_colors[self.instance_labels + 1]
        self.pcd_geom.colors = o3d.utility.Vector3dVector(point_colors)

        try:
            self.widget3d.scene.remove_geometry(POINT_CLOUD_NAME)
        except Exception as e:
            print(f"Info: Could not remove geometry '{POINT_CLOUD_NAME}' before update: {e}")

        self.widget3d.scene.add_geometry(POINT_CLOUD_NAME, self.pcd_geom, self.material)


        # 2. Update Centroid Markers
        if self.current_marker_names:
            for name in list(self.current_marker_names):
                try:
                    self.widget3d.scene.remove_geometry(name)
                except Exception as e:
                     print(f"Info: Could not remove marker geometry '{name}': {e}")
            self.current_marker_names.clear()

        marker_radius = self.args.marker_radius # Use radius from args
        marker_color = [1.0, 0.0, 0.0] # Red markers
        marker_material = rendering.MaterialRecord()
        marker_material.shader = "defaultLit"
        marker_material.base_color = [*marker_color, 1.0]

        if self.centroids:
            for inst_id, centroid in self.centroids.items():
                marker_name = f"{MARKER_BASE_NAME}{inst_id}"
                marker_geom = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
                marker_geom.translate(centroid)
                try:
                    self.widget3d.scene.add_geometry(marker_name, marker_geom, marker_material)
                    self.current_marker_names.add(marker_name)
                except Exception as e:
                     print(f"Error adding marker geometry '{marker_name}': {e}")

        # Update info label
        self.info_label.text = f"Instances Found: {num_instances_found}\nNoise Points: {num_noise_points}"


    def _on_close(self):
        # Handle window closing event
        print("Visualization window closed.")
        if self.args.save_results:
            self._save_current_results()
        self.window = None
        # Let run() handle application exit
        return True # Indicate window can close


    def _save_current_results(self):
        # Save results based on the *current* state of sliders/clustering
        if self.points_processed is None or self.instance_labels is None:
             print("Cannot save results, data not available.")
             return

        # Use the hardcoded output directory
        output_dir = HARDCODED_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        print(f"\nSaving results with timestamp {timestamp} to {output_dir}...")
        print(f"  Using final parameters: eps={self.current_eps:.4f}, min_samples={self.current_min_samples}")

        files_saved_msg = ["npy", "txt"]

        # Save processed points, semantic labels, and final instance labels
        points_save_path = os.path.join(output_dir, f"points_processed_{timestamp}.npy")
        sem_labels_save_path = os.path.join(output_dir, f"semantic_labels_{timestamp}.npy")
        inst_labels_save_path = os.path.join(output_dir, f"instance_labels_{timestamp}.npy")
        np.save(points_save_path, self.points_processed)
        np.save(sem_labels_save_path, self.semantic_labels)
        np.save(inst_labels_save_path, self.instance_labels)

        # Save combined TXT file (using processed points)
        data_to_save = np.hstack((
            self.points_processed,
            self.semantic_labels.reshape(-1, 1),
            self.instance_labels.reshape(-1, 1)
        ))
        txt_filename = os.path.join(output_dir, f"segmented_full_{timestamp}.txt")
        np.savetxt(txt_filename, data_to_save, fmt='%.6f,%.6f,%.6f,%d,%d', delimiter=',', header='x,y,z,semantic_label,instance_label')

        # Save individual instance PLY files (using processed points)
        num_ply_saved = 0
        if len(self.centroids) > 0 and OPEN3D_AVAILABLE:
            max_id = np.max(self.instance_labels) if len(self.unique_instances) > 0 else -1
            num_colors_needed = max_id + 2;
            num_colors_needed = max(num_colors_needed, len(self.centroids) + 1)
            rng_colors = np.random.default_rng(42); instance_colors = rng_colors.random((num_colors_needed, 3)); instance_colors[0] = [0.5, 0.5, 0.5]

            for inst_id in self.centroids.keys(): # Iterate through found instances
                 mask = (self.instance_labels == inst_id)
                 points_for_instance = self.points_processed[mask]
                 if points_for_instance.shape[0] == 0: continue

                 pcd_inst = o3d.geometry.PointCloud()
                 pcd_inst.points = o3d.utility.Vector3dVector(points_for_instance)
                 label_color = instance_colors[inst_id + 1] # Get color for this instance
                 pcd_inst.paint_uniform_color(label_color)

                 ply_filename = os.path.join(output_dir, f"instance_{timestamp}_id_{inst_id}.ply")
                 try:
                     o3d.io.write_point_cloud(ply_filename, pcd_inst, write_ascii=False)
                     num_ply_saved += 1
                 except Exception as e_ply:
                     print(f"\nError saving PLY for instance {inst_id}: {e_ply}")

            if num_ply_saved > 0: files_saved_msg.append(f"{num_ply_saved}_instance_ply")

        print(f"Saved output ({', '.join(files_saved_msg)})")


# --- Main Function (Modified for GUI and Hardcoded Paths) ---
def main_interactive(args):
    # Use hardcoded paths instead of args for these
    input_h5 = HARDCODED_INPUT_H5
    sample_index = HARDCODED_SAMPLE_INDEX
    checkpoint_path = HARDCODED_CHECKPOINT
    output_dir = HARDCODED_OUTPUT_DIR # Also used for saving check

    print(f"Starting interactive instance segmentation...")
    print(f"  Input HDF5: {input_h5}")
    print(f"  Sample Index: {sample_index}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output Dir: {output_dir}")
    print(f"Remaining Args: {args}") # Print the args that are still parsed

    # --- Dependency Checks ---
    if not SKLEARN_AVAILABLE: print("FATAL Error: scikit-learn required."); sys.exit(1)
    if not OPEN3D_AVAILABLE: print("FATAL Error: Open3D with GUI support required."); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"); print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True) # Use hardcoded output dir

    # --- Load Model (Once) ---
    print(f"Loading SEMANTIC model checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model_input_channels = 3 if args.no_rgb else 6
    print(f"Initializing model for {model_input_channels}D input features...")
    try:
        # Pass the parsed args object for model architecture params
        model = PyG_PointTransformerSegModel(num_classes=args.num_classes, args=args).to(device)
        print("Model structure initialized.")
    except Exception as e: print(f"Error init model: {e}"); return
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    except Exception as e: print(f"Error loading checkpoint file: {e}"); return
    try:
        if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint: model.load_state_dict(checkpoint['state_dict'])
        else: model.load_state_dict(checkpoint)
        print("Model weights loaded successfully.")
    except Exception as e: print(f"Error loading state dict: {e}"); return
    model.eval()

    # --- Load HDF5 Data (Once) ---
    print(f"Loading data from HDF5 file: {input_h5}")
    if not os.path.exists(input_h5): raise FileNotFoundError(f"Input HDF5 file not found: {input_h5}")

    points_xyz_np = None
    gt_labels_np = None # Ground truth labels (not used in interactive part, but loaded)
    points_rgb_np = None
    num_points_in_file = 0

    try:
        with h5py.File(input_h5, 'r') as f:
            required_keys = ['data', 'seg']
            if not args.no_rgb: required_keys.append('rgb')
            for key in required_keys:
                 if key not in f: raise KeyError(f"HDF5 file missing required key: '{key}'")

            dset_data = f['data']; dset_seg = f['seg']; dset_rgb = f.get('rgb')

            num_samples_in_file = dset_data.shape[0]; num_points_in_file = dset_data.shape[1]
            print(f"File contains {num_samples_in_file} samples, each with {num_points_in_file} points.")

            # Use args.num_points only for the initial check/warning if needed
            if num_points_in_file != args.num_points:
                 print(f"Warning: --num_points ({args.num_points}) does not match points in HDF5 ({num_points_in_file}). Using data as is.")
                 # We don't modify args here, just use num_points_in_file

            # Use hardcoded sample_index
            if not (0 <= sample_index < num_samples_in_file):
                raise IndexError(f"Sample index {sample_index} out of bounds (0 to {num_samples_in_file - 1}).")

            print(f"Extracting sample at index: {sample_index}")
            points_xyz_np = dset_data[sample_index].astype(np.float32) # (N, 3)
            gt_labels_np = dset_seg[sample_index].astype(np.int64)   # (N,)
            if dset_rgb is not None and not args.no_rgb:
                points_rgb_np = dset_rgb[sample_index].astype(np.uint8) # (N, 3) uint8

    except Exception as e:
        print(f"Error reading HDF5 file {input_h5}: {e}")
        return

    # --- Preprocess Data (Normalization + Feature Prep - Once) ---
    print("Preprocessing extracted sample...")
    points_processed_np = np.copy(points_xyz_np)

    # Use args.normalize
    if args.normalize:
        centroid = np.mean(points_processed_np, axis=0)
        points_processed_np = points_processed_np - centroid
        max_dist = np.max(np.sqrt(np.sum(points_processed_np ** 2, axis=1)))
        if max_dist > 1e-6: points_processed_np = points_processed_np / max_dist
        print("Applied normalization (center + unit sphere) to XYZ coordinates for processing.")
    else:
        print("Normalization disabled.")

    # Prepare features for the model
    features_list = [points_processed_np]
    if not args.no_rgb:
        if points_rgb_np is not None:
             rgb_normalized = points_rgb_np.astype(np.float32) / 255.0
             features_list.append(rgb_normalized)
        else:
             print("Warning: Model expects RGB (--no_rgb not set) but HDF5 had no 'rgb' key. Using default gray (0.5) features.")
             # Use num_points_in_file which reflects actual data size
             default_colors = np.full((num_points_in_file, 3), 0.5, dtype=np.float32)
             features_list.append(default_colors)

    features_np = np.concatenate(features_list, axis=1) # (N, 3) or (N, 6)

    if features_np.shape[1] != model_input_channels:
        print(f"FATAL ERROR: Model expects {model_input_channels}D features, but processed features have {features_np.shape[1]}D.")
        sys.exit(1)

    features_tensor = torch.from_numpy(features_np).float().unsqueeze(0).to(device) # (1, N, C)
    print(f"Preprocessing complete. Model input shape: {features_tensor.shape}")


    # --- Semantic Segmentation Inference (Once) ---
    print("Performing semantic segmentation inference...")
    with torch.no_grad():
        try:
            logits = model(features_tensor) # (1, N, num_classes)
            pred_semantic_labels_np = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy() # (N,)
            print("Semantic prediction complete.")
        except Exception as e:
            print(f"Error during model inference: {e}")
            print(f"Input tensor shape: {features_tensor.shape}")
            return

    # --- Initialize and Run the GUI Application ---
    print("\nInitializing interactive application...")
    app_instance = InstanceSegmenterApp(
        args, # Pass the remaining parsed args
        points_xyz_original=points_xyz_np,
        points_processed=points_processed_np,
        semantic_labels=pred_semantic_labels_np,
        input_h5_path=input_h5, # Pass hardcoded path for title
        sample_idx=sample_index # Pass hardcoded index for title
    )

    try:
        gui.Application.instance.run()
    except Exception as e:
        print(f"An error occurred during the application run: {e}")
    finally:
        print("\nInteractive instance segmentation finished.")


# --- Command Line Argument Parser (Reduced) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='INTERACTIVE instance segmentation using Open3D GUI with HARDCODED input paths.')

    # --- Data Processing (Still relevant) ---
    parser.add_argument('--num_points', type=int, default=204800, help='Expected number of points per sample (used for initial check).')
    parser.add_argument('--no_rgb', action='store_true', help="Assume model expects 3D input (ignores RGB even if present in HDF5).")
    parser.add_argument('--no_normalize', dest='normalize', action='store_false', default=True, help='Disable normalization (center + unit sphere) of loaded XYZ data before processing.')

    # --- Model Parameters (Required for model init) ---
    parser.add_argument('--num_classes', type=int, default=2, help='Number of segmentation classes model was trained for')
    parser.add_argument('--k_neighbors', type=int, default=16, help='(Model Arch) k for k-NN graph.')
    parser.add_argument('--embed_dim', type=int, default=64, help='(Model Arch) Initial embedding dimension.')
    parser.add_argument('--pt_hidden_dim', type=int, default=128, help='(Model Arch) Hidden dimension for PointTransformerConv.')
    parser.add_argument('--pt_heads', type=int, default=4, help='(Model Arch) Number of attention heads.')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='(Model Arch) Number of PointTransformerConv layers.')
    parser.add_argument('--dropout', type=float, default=0.3, help='(Model Arch) Dropout rate.')

    # --- Clustering Parameters (Initial Values for Sliders) ---
    parser.add_argument('--screw_label_id', type=int, default=1, help='The semantic label ID for the objects to be clustered (e.g., screws).')
    parser.add_argument('--dbscan_eps', type=float, default=0.05, help='INITIAL DBSCAN eps parameter. TUNABLE via slider.')
    parser.add_argument('--dbscan_min_samples', type=int, default=10, help='INITIAL DBSCAN min_samples parameter. TUNABLE via slider.')

    # --- Control Parameters (Still relevant) ---
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA.')
    parser.add_argument('--save_results', action='store_true', help='Save output arrays (NPY, TXT) and per-instance PLY files ON WINDOW CLOSE using final slider parameters.')
    parser.add_argument('--marker_radius', type=float, default=0.01, help='Radius of centroid markers in visualization (in processed coordinates).')

    args = parser.parse_args()

    # --- Final Checks ---
    if not SKLEARN_AVAILABLE: print("FATAL Error: scikit-learn required."); sys.exit(1)
    if not OPEN3D_AVAILABLE: print("FATAL Error: Open3D with GUI support required."); sys.exit(1)
    # Checks for hardcoded paths are done inside main_interactive now
    if not os.path.exists(HARDCODED_INPUT_H5): print(f"FATAL Error: Hardcoded Input HDF5 file not found: {HARDCODED_INPUT_H5}"); sys.exit(1)
    if not os.path.exists(HARDCODED_CHECKPOINT): print(f"FATAL Error: Hardcoded Checkpoint file not found: {HARDCODED_CHECKPOINT}"); sys.exit(1)


    # --- Run Main Interactive Function ---
    main_interactive(args)