# icp_configurator_ui.py
# 基于 PyQt5 的 ICP 姿态估计 UI 工具

import sys
import os
import datetime
import copy
import h5py
import numpy as np
import torch
import argparse
import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QFormLayout, QMessageBox, QListWidget
)
from PyQt5.QtCore import Qt, pyqtSignal

# --- 导入本地模块 ---
try:
    # Ensure your model.py is in the same directory or PYTHONPATH
    from model import PyG_PointTransformerSegModel # 语义分割模型类
except ImportError as e: print(f"FATAL Error importing model: {e}"); sys.exit(1)

# --- 导入 Open3D 和 Scikit-learn ---
try:
    import open3d as o3d
    from open3d.pipelines import registration as o3d_reg
    OPEN3D_AVAILABLE = True
except ImportError: print("FATAL Error: Open3D not found. Please install Open3D."); OPEN3D_AVAILABLE = False
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError: print("FATAL Error: scikit-learn not found. Please install scikit-learn."); SKLEARN_AVAILABLE = False


# --- 可视化函数: 显示 ICP 对齐结果 ---
def visualize_icp_alignment(source_pcd_transformed, target_pcd, window_name="ICP Alignment Result"):
    """可视化对齐后的源点云（实例）和目标点云（模型）。
       注意：这里 source_pcd_transformed 实际上是按照估计姿态变换后的目标模型点云。
    """
    if not OPEN3D_AVAILABLE: return
    # 给目标和源点云不同颜色以区分
    target_pcd_vis = copy.deepcopy(target_pcd) # 原始实例点云 (场景中的观察)
    source_pcd_transformed_vis = copy.deepcopy(source_pcd_transformed) # 按照估计姿态变换后的目标模型

    # 我们可视化 变换后的模型 (黄色) 与 原始实例 (蓝色)
    source_pcd_transformed_vis.paint_uniform_color([1, 0.706, 0]) # 黄色 (变换后的模型)
    target_pcd_vis.paint_uniform_color([0, 0.651, 0.929])  # 蓝色 (原始实例观察)


    print(f"\nDisplaying Final Pose Alignment for {window_name}...")
    print("Yellow: Estimated Model Pose | Blue: Observed Instance Points")
    print("(Close the window to continue...)")
    try:
        o3d.visualization.draw_geometries([source_pcd_transformed_vis, target_pcd_vis], window_name=window_name)
        print("Alignment visualization window closed.")
    except Exception as e:
        print(f"Error during visualization: {e}")
        QMessageBox.warning(None, "Visualization Error", f"Could not display visualization window: {e}")


class ICPConfiguratorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ICP Pose Estimation Configurator")
        self.setGeometry(100, 100, 800, 700) # Adjusted window size

        # --- Data Holders ---
        self.points_original_np = None
        self.target_pcd_original = None
        self.target_centroid_original = None
        self.instance_points_dict = {}
        self.current_instance_pcd_original = None # Store the original pcd for current instance
        self.current_estimated_pose = np.identity(4) # Store the latest calculated pose
        self.model_uses_rgb = False # <-- Added flag

        # --- UI Elements ---
        self.create_widgets()
        self.setup_layout()
        self.connect_signals()

        # --- Status ---
        self.data_loaded = False
        self.model_loaded = False
        self.model = None # Segmentation model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Store device
        print(f"Using device: {self.device}")


        # --- Dependency Check ---
        if not SKLEARN_AVAILABLE or not OPEN3D_AVAILABLE:
             QMessageBox.critical(self, "Dependencies Missing", "FATAL: scikit-learn or Open3D not found. Please install them.")
             self.setEnabled(False) # Disable UI if dependencies are missing


    def create_widgets(self):
        # File/Data Loading
        self.file_group = QGroupBox("Data Loading")
        self.file_layout = QFormLayout()
        self.h5_path_edit = QLineEdit("./data/my_custom_dataset_h5_rgb/test_0.h5") # Default value
        # Add browse buttons
        self.h5_browse_button = QPushButton("Browse...")
        self.h5_browse_button.clicked.connect(lambda: self.browse_file(self.h5_path_edit, "Select HDF5 File", "HDF5 Files (*.h5)"))

        self.model_path_edit = QLineEdit("stp/cube.STL") # Default value
        self.model_browse_button = QPushButton("Browse...")
        self.model_browse_button.clicked.connect(lambda: self.browse_file(self.model_path_edit, "Select Target Model File", "3D Model Files (*.ply *.stl *.obj)"))

        self.checkpoint_path_edit = QLineEdit("checkpoints_seg_pyg_ptconv_rgb/best_model.pth") # Default value
        self.checkpoint_browse_button = QPushButton("Browse...")
        self.checkpoint_browse_button.clicked.connect(lambda: self.browse_file(self.checkpoint_path_edit, "Select Model Checkpoint", "Checkpoint Files (*.pth)"))


        self.sample_index_spin = QSpinBox()
        self.sample_index_spin.setRange(0, 10000) # Adjust max range as needed
        self.target_label_spin = QSpinBox()
        self.target_label_spin.setRange(0, 255) # Assuming label IDs fit in uint8
        self.target_label_spin.setValue(1) # Default target label
        self.load_button = QPushButton("Load Data & Find Instances")

        # Add browse buttons to layout
        h5_layout = QHBoxLayout(); h5_layout.addWidget(self.h5_path_edit); h5_layout.addWidget(self.h5_browse_button)
        model_layout = QHBoxLayout(); model_layout.addWidget(self.model_path_edit); model_layout.addWidget(self.model_browse_button)
        checkpoint_layout = QHBoxLayout(); checkpoint_layout.addWidget(self.checkpoint_path_edit); checkpoint_layout.addWidget(self.checkpoint_browse_button)


        self.file_layout.addRow("HDF5 Path:", h5_layout)
        self.file_layout.addRow("Target Model Path:", model_layout)
        self.file_layout.addRow("Seg Checkpoint:", checkpoint_layout)
        self.file_layout.addRow("Sample Index:", self.sample_index_spin)
        self.file_layout.addRow("Target Label ID:", self.target_label_spin)
        self.file_layout.addRow(self.load_button)
        self.file_group.setLayout(self.file_layout)

        # Instance Selection
        self.instance_group = QGroupBox("Instances Found")
        self.instance_layout = QVBoxLayout()
        self.instance_list_widget = QListWidget()
        self.instance_layout.addWidget(QLabel("Select Instance:"))
        self.instance_layout.addWidget(self.instance_list_widget)
        self.instance_group.setLayout(self.instance_layout)
        self.instance_group.setEnabled(False) # Disabled until data is loaded

        # ICP Parameters
        self.icp_group = QGroupBox("ICP Parameters")
        self.icp_layout = QFormLayout()

        self.icp_threshold_spin = QDoubleSpinBox()
        self.icp_threshold_spin.setRange(0.1, 1000.0) # Adjust range based on your scale
        self.icp_threshold_spin.setSingleStep(0.1)
        self.icp_threshold_spin.setValue(20.0)
        self.icp_threshold_spin.setDecimals(3)

        self.icp_method_combo = QComboBox()
        self.icp_method_combo.addItems(["Point-to-Point", "Point-to-Plane"])

        self.icp_fitness_spin = QDoubleSpinBox()
        self.icp_fitness_spin.setRange(1e-9, 1.0) # Adjust range
        self.icp_fitness_spin.setSingleStep(1e-7)
        self.icp_fitness_spin.setValue(1e-6)
        self.icp_fitness_spin.setDecimals(9)

        self.icp_rmse_spin = QDoubleSpinBox()
        self.icp_rmse_spin.setRange(1e-9, 1.0) # Adjust range
        self.icp_rmse_spin.setSingleStep(1e-7)
        self.icp_rmse_spin.setValue(1e-6)
        self.icp_rmse_spin.setDecimals(9)

        self.icp_max_iter_spin = QSpinBox()
        self.icp_max_iter_spin.setRange(1, 10000)
        self.icp_max_iter_spin.setValue(1000)

        self.icp_min_points_spin = QSpinBox()
        self.icp_min_points_spin.setRange(1, 10000)
        self.icp_min_points_spin.setValue(50)


        self.icp_layout.addRow("Threshold:", self.icp_threshold_spin)
        self.icp_layout.addRow("Estimation Method:", self.icp_method_combo)
        self.icp_layout.addRow("Rel Fitness:", self.icp_fitness_spin)
        self.icp_layout.addRow("Rel RMSE:", self.icp_rmse_spin)
        self.icp_layout.addRow("Max Iterations:", self.icp_max_iter_spin)
        self.icp_layout.addRow("Min Points:", self.icp_min_points_spin)

        self.run_icp_button = QPushButton("Run ICP for Selected Instance")
        self.icp_layout.addRow(self.run_icp_button)
        self.icp_group.setLayout(self.icp_layout)
        self.icp_group.setEnabled(False) # Disabled until instance is selected

        # Results Display
        self.results_group = QGroupBox("Results")
        self.results_layout = QVBoxLayout()
        self.fitness_label = QLabel("Fitness: N/A")
        self.rmse_label = QLabel("RMSE: N/A")
        self.pose_label = QLabel("Pose Matrix: N/A")
        self.pose_label.setWordWrap(True) # Wrap long text

        self.results_layout.addWidget(self.fitness_label)
        self.results_layout.addWidget(self.rmse_label)
        self.results_layout.addWidget(self.pose_label)

        self.visualize_button = QPushButton("Visualize Alignment")
        self.results_layout.addWidget(self.visualize_button)

        self.results_group.setLayout(self.results_layout)
        self.results_group.setEnabled(False) # Disabled until ICP is run

    def setup_layout(self):
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.file_group)

        instance_icp_layout = QHBoxLayout()
        instance_icp_layout.addWidget(self.instance_group)
        instance_icp_layout.addWidget(self.icp_group)

        main_layout.addLayout(instance_icp_layout)
        main_layout.addWidget(self.results_group)

        self.setLayout(main_layout)

    def connect_signals(self):
        self.load_button.clicked.connect(self.load_data)
        self.instance_list_widget.currentRowChanged.connect(self.instance_selected)
        self.run_icp_button.clicked.connect(self.run_icp)
        self.visualize_button.clicked.connect(self.visualize_result)

    def browse_file(self, line_edit, caption, filter):
        """Helper function to open a file dialog and set the selected file path."""
        filePath, _ = QFileDialog.getOpenFileName(self, caption, "", filter)
        if filePath:
            line_edit.setText(filePath)


    def load_data(self):
        if not SKLEARN_AVAILABLE or not OPEN3D_AVAILABLE:
             QMessageBox.critical(self, "Error", "Dependencies missing.")
             return

        h5_path = self.h5_path_edit.text()
        model_path = self.model_path_edit.text()
        checkpoint_path = self.checkpoint_path_edit.text()
        sample_index = self.sample_index_spin.value()
        target_label_id = self.target_label_spin.value()

        if not os.path.exists(h5_path):
            QMessageBox.critical(self, "Error", f"HDF5 file not found: {h5_path}")
            return
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", f"Model file not found: {model_path}")
            return
        if not os.path.exists(checkpoint_path):
             QMessageBox.critical(self, "Error", f"Checkpoint file not found: {checkpoint_path}")
             return

        # --- Reset previous data ---
        self.points_original_np = None
        self.target_pcd_original = None
        self.target_centroid_original = None
        self.instance_points_dict = {}
        self.current_instance_pcd_original = None
        self.current_estimated_pose = np.identity(4)
        self.instance_list_widget.clear()
        self.icp_group.setEnabled(False)
        self.results_group.setEnabled(False)
        self.fitness_label.setText("Fitness: N/A")
        self.rmse_label.setText("RMSE: N/A")
        self.pose_label.setText("Pose Matrix: N/A")
        self.model = None # Reset model to force reload if checkpoint changes

        try:
            # --- Load and Preprocess HDF5 Data ---
            print(f"Loading data from HDF5: {h5_path}, Sample: {sample_index}")
            with h5py.File(h5_path, 'r') as f:
                if 'data' not in f: raise KeyError("'data' key missing")
                dset_data = f['data']
                dset_rgb = f['rgb'] if 'rgb' in f else None # Check for rgb data

                if not (0 <= sample_index < dset_data.shape[0]): raise IndexError("Sample index out of bounds.")
                self.points_original_np = dset_data[sample_index].astype(np.float32) # Load original points
                points_rgb_np = dset_rgb[sample_index].astype(np.uint8) if dset_rgb is not None else None

                # Determine if the model should use RGB based on the HDF5 data
                # Assuming if RGB data exists in H5, the model was trained with it.
                self.model_uses_rgb = (dset_rgb is not None)


            # --- Semantic Segmentation Inference ---
            # Simplified model loading and inference for UI
            # Load model only once per checkpoint path
            # Use a flag to track if model was loaded assuming RGB input
            print(f"Initializing model for {'XYZ+RGB' if self.model_uses_rgb else 'XYZ'} input...")

            # Create dummy args needed by PyG_PointTransformerSegModel constructor
            # These control the model's architecture, not the input tensor shape directly
            model_args = argparse.Namespace()
            model_args.num_classes = 2 # Assuming 2 classes based on original script default
            model_args.embed_dim = 64
            model_args.k_neighbors = 16
            model_args.pt_hidden_dim = 128
            model_args.pt_heads = 4
            model_args.num_transformer_layers = 2
            model_args.dropout = 0.3
            # Pass the determined RGB usage to the model args, even if model doesn't store it
            model_args.no_rgb = not self.model_uses_rgb


            try: self.model = PyG_PointTransformerSegModel(num_classes=model_args.num_classes, args=model_args).to(self.device); print("Model structure initialized.")
            except Exception as e: raise RuntimeError(f"Error init model: {e}")
            try: checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            except Exception as e: raise RuntimeError(f"Error loading checkpoint file: {e}")
            try:
                if 'model_state_dict' in checkpoint: self.model.load_state_dict(checkpoint['model_state_dict'])
                else: self.model.load_state_dict(checkpoint)
                print("Model weights loaded successfully.")
                self.model_loaded = True
            except Exception as e: raise RuntimeError(f"Error loading state dict: {e}")

            self.model.eval()


            print("\nPerforming semantic segmentation inference...")
            # Prepare model input - using original points and adding RGB if available, based on self.model_uses_rgb
            features_list = [self.points_original_np]
            if self.model_uses_rgb:
                 if points_rgb_np is not None:
                    features_list.append(points_rgb_np.astype(np.float32) / 255.0)
                    print("Using XYZ+RGB features for inference (RGB data found and model expects it).")
                 else:
                    # This case shouldn't happen if model_uses_rgb is based on dset_rgb,
                    # but as a fallback, pad if dset_rgb existed but current sample lacks it.
                    features_list.append(np.full((self.points_original_np.shape[0], 3), 0.5, dtype=np.float32))
                    print("Using XYZ + dummy RGB features for inference (RGB data expected but missing for this sample).")
            else:
                 print("Using XYZ features for inference (model expects 3D or no RGB data found in H5).")

            features_np = np.concatenate(features_list, axis=1)
            features_tensor = torch.from_numpy(features_np).float().unsqueeze(0).to(self.device)

            # Add a check to ensure feature dimension matches model expectation
            expected_feature_dim = 6 if self.model_uses_rgb else 3
            if features_np.shape[1] != expected_feature_dim:
                 raise ValueError(f"Prepared features dimension ({features_np.shape[1]}D) does not match model expected dimension ({expected_feature_dim}D). Check HDF5 data and model input config.")


            with torch.no_grad(): logits = self.model(features_tensor)
            pred_semantic_labels_np = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy() # (N,)
            print("Semantic prediction complete.")
            unique_semantic, counts_semantic = np.unique(pred_semantic_labels_np, return_counts=True)
            print(f"  Predicted semantic label distribution: {dict(zip(unique_semantic, counts_semantic))}")


            # --- DBSCAN Clustering ---
            print("\nPerforming DBSCAN clustering...")
            target_mask = (pred_semantic_labels_np == target_label_id)
            target_points_xyz = self.points_original_np[target_mask] # Use original points for clustering

            dbscan_eps = 50 # Using default from previous script
            dbscan_min_samples = 1000 # Using default from previous script

            if target_points_xyz.shape[0] >= dbscan_min_samples:
                db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, n_jobs=-1)
                instance_labels_target_points = db.fit_predict(target_points_xyz)
                unique_instances = np.unique(instance_labels_target_points[instance_labels_target_points != -1])
                unique_instances.sort()
                num_instances_found = len(unique_instances)
                print(f"Clustering found {num_instances_found} potential instances with label {target_label_id}.")
                self.instance_points_dict = {}
                self.instance_list_widget.clear() # Clear previous list
                for inst_id in unique_instances:
                    instance_mask_local = (instance_labels_target_points == inst_id)
                    self.instance_points_dict[inst_id] = target_points_xyz[instance_mask_local]
                    self.instance_list_widget.addItem(f"Instance {inst_id} ({self.instance_points_dict[inst_id].shape[0]} points)")
                    print(f"  Instance {inst_id}: {self.instance_points_dict[inst_id].shape[0]} points")

                if num_instances_found > 0:
                    self.instance_group.setEnabled(True)
                    self.instance_list_widget.setCurrentRow(0) # Select the first instance by default
                else:
                     print("No instances found after clustering.")
                     self.instance_group.setEnabled(False)
                     self.icp_group.setEnabled(False)
                     self.results_group.setEnabled(False)


            else:
                print(f"Not enough points predicted as target (label {target_label_id}) for DBSCAN.")
                self.instance_group.setEnabled(False)
                self.icp_group.setEnabled(False)
                self.results_group.setEnabled(False)


            # --- Load Target Model ---
            print(f"\nLoading target model file from: {model_path}")
            target_mesh = o3d.io.read_triangle_mesh(model_path)
            if not target_mesh.has_vertices():
                 print("  Could not load as mesh or mesh is empty, attempting to load as point cloud...")
                 self.target_pcd_original = o3d.io.read_point_cloud(model_path)
                 if not self.target_pcd_original.has_points(): raise ValueError("Target model file contains no points.")
                 self.target_centroid_original = self.target_pcd_original.get_center()
            else:
                 print(f"  Loaded mesh with {len(target_mesh.vertices)} vertices. Sampling points...")
                 mesh_bbox = target_mesh.get_axis_aligned_bounding_box()
                 self.target_centroid_original = mesh_bbox.get_center()
                 num_model_points_to_sample = max(self.points_original_np.shape[0] * 2, 8192) # Sample more points than input
                 self.target_pcd_original = target_mesh.sample_points_uniformly(number_of_points=num_model_points_to_sample)

            print(f"Target model loaded/sampled as point cloud with {len(self.target_pcd_original.points)} points.")
            print(f"  Original Target Model Center: [{self.target_centroid_original[0]:.3f},{self.target_centroid_original[1]:.3f},{self.target_centroid_original[2]:.3f}]")
            self.data_loaded = True
            QMessageBox.information(self, "Success", "Data, Model, Segmentation, and Clustering loaded successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error during Loading", f"An error occurred: {e}")
            print(f"Error during loading: {e}")
            self.data_loaded = False
            self.model_loaded = False
            self.model = None # Ensure model is reset on error
            self.instance_group.setEnabled(False)
            self.icp_group.setEnabled(False)
            self.results_group.setEnabled(False)


    def instance_selected(self, row):
        if row < 0 or row >= self.instance_list_widget.count(): # No instance selected or invalid row
            self.icp_group.setEnabled(False)
            self.results_group.setEnabled(False)
            self.current_instance_pcd_original = None
            return

        # Get instance ID from the list widget item text (assuming format "Instance ID (...)")
        item_text = self.instance_list_widget.item(row).text()
        try:
            # Extract integer ID from string like "Instance 123 (456 points)"
            instance_id_str = item_text.split(' ')[1]
            instance_id = int(instance_id_str)
        except (ValueError, IndexError):
            print(f"Could not parse instance ID from item text: {item_text}")
            self.icp_group.setEnabled(False)
            self.results_group.setEnabled(False)
            self.current_instance_pcd_original = None
            return


        if instance_id not in self.instance_points_dict:
             print(f"Instance ID {instance_id} not found in instance_points_dict.")
             self.icp_group.setEnabled(False)
             self.results_group.setEnabled(False)
             self.current_instance_pcd_original = None
             return


        instance_points_np = self.instance_points_dict[instance_id]
        print(f"\nInstance {instance_id} selected with {instance_points_np.shape[0]} points.")

        self.current_instance_pcd_original = o3d.geometry.PointCloud()
        self.current_instance_pcd_original.points = o3d.utility.Vector3dVector(instance_points_np)

        # Enable ICP controls
        self.icp_group.setEnabled(True)
        self.results_group.setEnabled(False) # Disable results until ICP is run
        self.fitness_label.setText("Fitness: N/A")
        self.rmse_label.setText("RMSE: N/A")
        self.pose_label.setText("Pose Matrix: N/A")


    def run_icp(self):
        if not self.data_loaded or not self.model_loaded or self.current_instance_pcd_original is None:
            QMessageBox.warning(self, "Warning", "Please load data and select an instance first.")
            return
        if not OPEN3D_AVAILABLE:
             QMessageBox.critical(self, "Error", "Open3D not available.")
             return

        # --- Get ICP Parameters from UI ---
        threshold = self.icp_threshold_spin.value()
        method_str = self.icp_method_combo.currentText()
        relative_fitness = self.icp_fitness_spin.value()
        relative_rmse = self.icp_rmse_spin.value() # Typo fixed
        max_iteration = self.icp_max_iter_spin.value()
        min_points = self.icp_min_points_spin.value()

        if len(self.current_instance_pcd_original.points) < min_points:
             QMessageBox.warning(self, "Warning", f"Selected instance has only {len(self.current_instance_pcd_original.points)} points, which is less than the minimum required ({min_points}). Skipping ICP.")
             self.results_group.setEnabled(False)
             return

        # --- Prepare Point Clouds for ICP ---
        # ICP will be performed on centered point clouds
        # Use a copy of the original point clouds to avoid modifying them persistently

        # Target Model (centered) - using a fresh centered copy for safety
        target_pcd_centered = copy.deepcopy(self.target_pcd_original)
        target_pcd_centered.translate(-self.target_centroid_original)
        print(f"\nPrepared Target Model for ICP (centered). Points: {len(target_pcd_centered.points)}")

        # Current Instance (centered) - using a copy of the currently selected original instance pcd
        source_pcd_original = copy.deepcopy(self.current_instance_pcd_original)
        source_centroid_original = source_pcd_original.get_center() # Get original instance center
        source_pcd_centered = copy.deepcopy(source_pcd_original) # Create centered copy for ICP
        source_pcd_centered.translate(-source_centroid_original)
        print(f"Prepared Source Instance for ICP (centered). Points: {len(source_pcd_centered.points)}. Original Center: {source_centroid_original}")


        # --- Configure ICP ---
        estimation_method = None
        if method_str == 'Point-to-Point':
            estimation_method = o3d_reg.TransformationEstimationPointToPoint()
            print("Using Point-to-Point ICP.")
        elif method_str == 'Point-to-Plane':
            print("Using Point-to-Plane ICP.")
            # Need to compute normals for Point-to-Plane
            print("  Estimating normals for source (instance) and target (model)...")
            # Use a radius relevant to the point cloud density/scale, perhaps related to ICP threshold
            # Radius and max_nn should be tuned based on your data's resolution
            normal_radius = threshold * 2.0 # Example radius based on threshold
            normal_radius = max(1e-3, normal_radius) # Ensure radius is positive

            try:
                # Always try to estimate normals if Point-to-Plane is selected
                source_pcd_centered.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
                target_pcd_centered.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
                # Optional: Orient normals - might be needed for consistency, can be complex for partial clouds
                # source_pcd_centered.orient_normals_consistent_tangent_plane(k=30)
                # target_pcd_centered.orient_normals_consistent_tangent_plane(k=30)

                # Check if normals were successfully created
                if source_pcd_centered.has_normals() and target_pcd_centered.has_normals():
                     estimation_method = o3d_reg.TransformationEstimationPointToPlane()
                     print("  Normals estimated successfully for Point-to-Plane.")
                else:
                     print("  Normal estimation failed or resulted in no normals. Switching to Point-to-Point.")
                     estimation_method = o3d_reg.TransformationEstimationPointToPoint()
                     QMessageBox.warning(self, "Warning", "Normal estimation failed for Point-to-Plane ICP. Switching to Point-to-Point.")

            except Exception as e:
                 print(f"Error during normal estimation: {e}")
                 print("Switching to Point-to-Point ICP.")
                 estimation_method = o3d_reg.TransformationEstimationPointToPoint()
                 QMessageBox.warning(self, "Warning", f"Error during normal estimation for Point-to-Plane ICP ({e}). Switching to Point-to-Point.")


        if estimation_method is None: # Should not happen if method_str is one of the choices, but as a safeguard
            print("FATAL ERROR: Could not determine ICP estimation method.")
            QMessageBox.critical(self, "Error", "Could not determine ICP estimation method.")
            self.results_group.setEnabled(False)
            return


        criteria = o3d_reg.ICPConvergenceCriteria(
             relative_fitness=relative_fitness,
             relative_rmse=relative_rmse,
             max_iteration=max_iteration
        )
        print(f"ICP Parameters: Threshold={threshold}, Method={method_str}, FitCrit={relative_fitness}, RMSE crit={relative_rmse}, MaxIter={max_iteration}")

        # Initial transform: Using identity matrix for centered point clouds
        # TODO: Integrate global registration here for a better initial guess
        initial_transform = np.identity(4)
        print("Using Identity matrix as initial transform for ICP.")


        # --- Execute ICP ---
        print("Executing ICP...")
        try:
            start_time = time.time()
            reg_result = o3d_reg.registration_icp(
                source_pcd_centered, # Centered instance point cloud
                target_pcd_centered, # Centered model point cloud
                threshold,
                initial_transform, # Initial guess (identity for now)
                estimation_method,
                criteria
            )
            end_time = time.time()
            print(f"ICP finished in {end_time - start_time:.3f} seconds.")

            print(f"ICP Result: Fitness={reg_result.fitness:.4f}, RMSE={reg_result.inlier_rmse:.4f}")
            print("Estimated Transformation Matrix (source_centered -> target_centered):")
            print(reg_result.transformation)

            # --- Calculate Final Pose (Target_original -> Source_original) ---
            # ICP provides T_ICP, which transforms source_centered to target_centered:
            # T_ICP * (S_orig - Cs_orig) ≈ (T_orig - Ct_original)
            #
            # We want Pose P, which transforms T_orig to S_orig:
            # S_orig ≈ P * T_orig
            #
            # Final Pose P = T_from_cent_S_orig * T_ICP_inv * T_to_cent_T_orig
            # P = [ I | Cs_orig ] * [ R_ICP_inv | -R_ICP_inv * t_ICP ] * [ I | -Ct_original ]
            # P = [ R_ICP_inv | Cs_orig - R_ICP_inv * Ct_original - R_ICP_inv * t_ICP ]
            #
            # Where:
            # T_ICP is reg_result.transformation
            # R_ICP is T_ICP[:3,:3], t_ICP is T_ICP[:3,3]
            # Cs_orig is source_centroid_original (original instance center)
            # Ct_original is self.target_centroid_original (original target model center)

            T_icp = reg_result.transformation
            R_icp = T_icp[:3, :3]
            t_icp = T_icp[:3, 3]

            R_icp_inv = np.linalg.inv(R_icp)

            Cs_orig = source_centroid_original # Original instance center
            Ct_original = self.target_centroid_original # Original target model center

            # Calculate final translation vector using the corrected formula
            # t_final_corrected = Cs_orig - R_icp_inv @ Ct_original - R_icp_inv @ t_icp
            t_final_corrected = Cs_orig - np.dot(R_icp_inv, Ct_original) - np.dot(R_icp_inv, t_icp)

            self.current_estimated_pose = np.identity(4)
            self.current_estimated_pose[:3, :3] = R_icp_inv
            self.current_estimated_pose[:3, 3] = t_final_corrected

            print("\nCalculated Final Pose Matrix (Target_original -> Source_original):")
            print(self.current_estimated_pose)

            # --- Display Results ---
            self.fitness_label.setText(f"Fitness: {reg_result.fitness:.6f}")
            self.rmse_label.setText(f"RMSE: {reg_result.inlier_rmse:.6f}")
            # Format pose matrix nicely for display
            pose_str = np.array2string(self.current_estimated_pose, separator=', ', max_line_width=100, formatter={'float_kind':lambda x: "%.4f" % x})
            self.pose_label.setText(f"Pose Matrix:\n{pose_str}")

            self.results_group.setEnabled(True)
            QMessageBox.information(self, "ICP Complete", "ICP calculation finished. Check results and visualize.")

        except Exception as e:
             QMessageBox.critical(self, "Error during ICP", f"An error occurred during ICP: {e}")
             print(f"Error during ICP: {e}")
             self.results_group.setEnabled(False)
             self.fitness_label.setText("Fitness: Error")
             self.rmse_label.setText("RMSE: Error")
             self.pose_label.setText("Pose Matrix: Error")
             self.current_estimated_pose = np.identity(4)


    def visualize_result(self):
        if not self.data_loaded or not self.model_loaded or self.current_instance_pcd_original is None:
            QMessageBox.warning(self, "Warning", "Please load data and select an instance first.")
            return
        # Check if the pose is different from identity (meaning ICP was likely run)
        # Allow visualizing even if fitness/RMSE is bad
        # if np.array_equal(self.current_estimated_pose, np.identity(4)):
        #      QMessageBox.warning(self, "Warning", "Please run ICP first to get a valid pose.")
        #      return
        if not OPEN3D_AVAILABLE:
             QMessageBox.critical(self, "Error", "Open3D not available.")
             return

        print("\nVisualizing alignment...")
        # Transform the original target model using the calculated pose
        target_pcd_transformed_by_pose = copy.deepcopy(self.target_pcd_original)
        target_pcd_transformed_by_pose.transform(self.current_estimated_pose)

        # Visualize the transformed target model and the original instance points
        # Using the helper function from before
        visualize_icp_alignment(
            target_pcd_transformed_by_pose, # Transformed target model (yellow)
            self.current_instance_pcd_original, # Original instance observation (blue)
            window_name="Final Pose Alignment"
        )


    def closeEvent(self, event):
        # Clean up resources if necessary
        print("Closing application.")
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QMainWindow() # Use QMainWindow for a more standard window layout if needed
    ui = ICPConfiguratorUI()
    main_window.setCentralWidget(ui)
    main_window.setWindowTitle("ICP Pose Estimation Configurator")
    main_window.show()
    sys.exit(app.exec_())