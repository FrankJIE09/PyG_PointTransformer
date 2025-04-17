import open3d as o3d
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and display a mesh file (PLY, STL, OBJ, etc.).")
    parser.add_argument("--mesh_file", type=str, required=True, help="Path to the mesh file.")
    args = parser.parse_args()

    if not os.path.exists(args.mesh_file):
        print(f"Error: File not found at '{args.mesh_file}'")
    else:
        print(f"Loading mesh file: {args.mesh_file}")
        try:
            # 尝试作为三角网格加载
            mesh = o3d.io.read_triangle_mesh(args.mesh_file)
            if not mesh.has_vertices():
                 # 如果作为网格加载失败或为空，尝试作为点云加载（某些PLY可能是点云格式）
                 print("  Could not load as mesh or mesh is empty, trying as point cloud...")
                 pcd = o3d.io.read_point_cloud(args.mesh_file)
                 if not pcd.has_points():
                     print("Error: Failed to load the file as mesh or point cloud.")
                 else:
                      print(f"Loaded as point cloud with {len(pcd.points)} points.")
                      print("Displaying point cloud...")
                      o3d.visualization.draw_geometries([pcd], window_name=os.path.basename(args.mesh_file))
            else:
                print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
                # 计算法线以便更好地着色显示
                if not mesh.has_vertex_normals():
                     mesh.compute_vertex_normals()
                print("Displaying mesh...")
                o3d.visualization.draw_geometries([mesh], window_name=os.path.basename(args.mesh_file))

        except Exception as e:
            print(f"Error processing file: {e}")

        print("Visualization window closed.")