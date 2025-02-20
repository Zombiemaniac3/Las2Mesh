#!/usr/bin/env python3
import sys
import open3d as o3d

def main():
    if len(sys.argv) != 2:
        print("Usage: python stl_viewer_fast.py <file.stl>")
        sys.exit(1)

    stl_file = sys.argv[1]
    mesh = o3d.io.read_triangle_mesh(stl_file)
    if mesh.is_empty():
        print("Error: Mesh is empty or failed to load.")
        sys.exit(1)

    # Compute normals for proper shading
    mesh.compute_vertex_normals()

    # Center the mesh so its center is at the origin
    center = mesh.get_center()
    mesh.translate(-center)

    # Optional: If the mesh is extremely dense (tens of millions of triangles),
    # you can simplify it to improve interactivity.
    # Uncomment and adjust the target number if needed:
    # mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=5000000)

    # Create an Open3D visualizer window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Fast STL Viewer", width=1280, height=720)
    vis.add_geometry(mesh)

    # Adjust the view so the model is centered
    view_ctl = vis.get_view_control()
    #view_ctl.reset_camera_to_default()

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()

