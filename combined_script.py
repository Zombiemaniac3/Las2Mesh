#!/usr/bin/env python3
"""
LAS to STL and optional STL to LandXML/PCD Converter with a dynamic UI,
plus integrated STL viewer (via Open3D)
"""

import os
import sys
import ctypes
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

import laspy
import numpy as np
from scipy.spatial import Delaunay
import trimesh
from tqdm import tqdm
from stl import mesh
from dask.distributed import Client
from tkinter import ttk  # Add this import at the top


try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False

try:
    import open3d as o3d
except ImportError:
    o3d = None

####################################
# Simple tooltip class for Tkinter widgets
####################################
class CreateToolTip(object):
    """
    Create a tooltip for a given widget.
    """
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        widget.bind("<Enter>", self.enter)
        widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)

    def unschedule(self):
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)

    def showtip(self, event=None):
        if self.tipwindow:
            return
        # calculate position for the tooltip window
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 25
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"), wraplength=300)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

####################################
# Minimal LandXML template
####################################
LANDXML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<LandXML version="1.2" xmlns="http://www.landxml.org/schema/LandXML-1.2">
  <Application name="MyApp" desc="LandXML Export" manufacturer="MyCompany" version="1.0" manufacturerURL="http://www.example.com" />
  <Units>
    <Metric linearUnit="meter" elevationUnit="meter" />
  </Units>
  <Surfaces>
    <Surface name="{surface_name}">
      <Definition surfType="TIN">
        <Pnts>
          {points_block}
        </Pnts>
        <Faces>
          {faces_block}
        </Faces>
      </Definition>
    </Surface>
  </Surfaces>
</LandXML>
"""

####################################
# C adaptive decimation via ctypes
####################################
script_dir = os.path.dirname(os.path.abspath(__file__))

####################################
# Parallel triangulation utilities using Dask
####################################
def process_tile_dask(tile, all_points):
    expanded_bbox = tile['expanded']
    interior_bbox = tile['interior']
    bx0, by0, bx1, by1 = expanded_bbox
    mask = ((all_points[:, 0] >= bx0) & (all_points[:, 0] <= bx1) &
            (all_points[:, 1] >= by0) & (all_points[:, 1] <= by1))
    tile_points = all_points[mask]
    if tile_points.shape[0] < 3:
        return None
    try:
        tri = Delaunay(tile_points[:, :2])
    except Exception:
        return None
    faces = tri.simplices
    verts, faces = keep_partial_in_interior(tile_points, faces, interior_bbox)
    if verts is None or faces is None:
        return None
    return (verts, faces)

def keep_partial_in_interior(verts, faces, bbox):
    (bx0, by0, bx1, by1) = bbox
    inside_mask = ((verts[:, 0] >= bx0) & (verts[:, 0] <= bx1) &
                   (verts[:, 1] >= by0) & (verts[:, 1] <= by1))
    face_indices = []
    for i, f in enumerate(faces):
        if inside_mask[f[0]] or inside_mask[f[1]] or inside_mask[f[2]]:
            face_indices.append(i)
    face_indices = np.array(face_indices, dtype=np.int64)
    if len(face_indices) == 0:
        return (None, None)
    new_faces = faces[face_indices]
    used_verts = np.unique(new_faces.flatten())
    mapping = -np.ones(verts.shape[0], dtype=np.int64)
    mapping[used_verts] = np.arange(len(used_verts))
    new_faces = mapping[new_faces]
    new_verts = verts[used_verts]
    return (new_verts, new_faces)

####################################
# LAS -> STL (and optionally LAS) converter
####################################
def convert_las_to_stl(input_las, output_stl,
                       adaptive_grid=None, adaptive_minfrac=0.1,
                       adaptive_maxfrac=1.0, curve_exponent=1.0,
                       flat_threshold=0.0, flat_fraction=0.0,
                       curb_threshold=0.2, use_gpu_decimation=False,
                       output_xyz=None):
    with laspy.open(input_las) as las_file:
        total_points = las_file.header.point_count
        chunk_size = 500_000
        points_list = []
        with tqdm(total=total_points, desc="Reading LAS", unit="pts") as pbar:
            for las_chunk in las_file.chunk_iterator(chunk_size):
                x, y, z = las_chunk.x, las_chunk.y, las_chunk.z
                chunk_points = np.vstack((x, y, z)).T
                points_list.append(chunk_points)
                pbar.update(len(x))
        all_points = np.concatenate(points_list, axis=0)

    # Optional adaptive decimation
    if adaptive_grid is not None and adaptive_grid > 1:
        print("Adaptive decimation settings:")
        print(f"  grid={adaptive_grid}")
        print(f"  minFrac={adaptive_minfrac}, maxFrac={adaptive_maxfrac}")
        print(f"  curveExponent={curve_exponent}")
        print(f"  flatThreshold={flat_threshold}, flatFraction={flat_fraction}")
        print(f"  curbEdgeThreshold={curb_threshold} m")
        before_count = len(all_points)
        n_points = all_points.shape[0]
        points_arr = np.ascontiguousarray(all_points, dtype=np.double)

        if use_gpu_decimation:
            if os.name == "nt":
                dll_path = os.path.join(script_dir, "decimation_cuda.dll")
            else:
                dll_path = os.path.join(script_dir, "libdecimation_cuda.so")
            try:
                lib = ctypes.CDLL(dll_path)
            except OSError as e:
                sys.exit(f"Could not load GPU decimation library:\n{dll_path}\nError: {e}")
            func = lib.adaptive_decimate_points_cuda
        else:
            if os.name == "nt":
                dll_path = os.path.join(script_dir, "decimation.dll")
            else:
                dll_path = os.path.join(script_dir, "libdecimation.so")
            try:
                lib = ctypes.CDLL(dll_path)
            except OSError as e:
                sys.exit(f"Could not load CPU decimation library:\n{dll_path}\nError: {e}")
            func = lib.adaptive_decimate_points

        func.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.POINTER(ctypes.c_int)
        ]
        func.restype = ctypes.c_int

        out_points = ctypes.POINTER(ctypes.c_double)()
        out_n_points = ctypes.c_int(0)
        ret = func(
            points_arr,
            n_points,
            adaptive_grid,
            adaptive_minfrac,
            adaptive_maxfrac,
            curve_exponent,
            flat_threshold,
            flat_fraction,
            curb_threshold,
            ctypes.byref(out_points),
            ctypes.byref(out_n_points)
        )
        if ret != 0:
            raise Exception("Adaptive decimation failed in C/CUDA code")
        kept_count = out_n_points.value
        decimated_points = np.ctypeslib.as_array(out_points, shape=(kept_count, 3)).copy()
        if os.name != "nt":
            libc = ctypes.CDLL("libc.so.6")
            libc.free.argtypes = [ctypes.c_void_p]
            libc.free(ctypes.cast(out_points, ctypes.c_void_p))
        after_count = decimated_points.shape[0]
        print(f"Retained {after_count} / {before_count} points after adaptive decimation.")
        all_points = decimated_points

    # Optional: export the (decimated) point cloud as LAS
    if output_xyz is not None and output_xyz.strip():
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.x_offset = np.min(all_points[:, 0])
        header.y_offset = np.min(all_points[:, 1])
        header.z_offset = np.min(all_points[:, 2])
        header.x_scale = 0.001
        header.y_scale = 0.001
        header.z_scale = 0.001

        las = laspy.LasData(header)
        las.x = all_points[:, 0]
        las.y = all_points[:, 1]
        las.z = all_points[:, 2]
        las.write(output_xyz)
        print(f"Successfully exported decimated point cloud to LAS: {output_xyz}")

    # Tiling and triangulation
    min_x, min_y = np.min(all_points[:, :2], axis=0)
    max_x, max_y = np.max(all_points[:, :2], axis=0)
    tile_count_x = 4
    tile_count_y = 4
    overlap_ratio = 0.1
    range_x = max_x - min_x
    range_y = max_y - min_y
    tile_width = range_x / tile_count_x
    tile_height = range_y / tile_count_y
    overlap_x = tile_width * overlap_ratio
    overlap_y = tile_height * overlap_ratio

    tiles = []
    for i in range(tile_count_x):
        for j in range(tile_count_y):
            tile_min_x = min_x + i * tile_width
            tile_min_y = min_y + j * tile_height
            tile_max_x = tile_min_x + tile_width
            tile_max_y = tile_min_y + tile_height
            interior_bbox = (tile_min_x, tile_min_y, tile_max_x, tile_max_y)
            tile_min_x_ov = tile_min_x - overlap_x if i > 0 else tile_min_x
            tile_min_y_ov = tile_min_y - overlap_y if j > 0 else tile_min_y
            tile_max_x_ov = tile_max_x + overlap_x if i < (tile_count_x - 1) else tile_max_x
            tile_max_y_ov = tile_max_y + overlap_y if j < (tile_count_y - 1) else tile_max_y
            expanded_bbox = (tile_min_x_ov, tile_min_y_ov, tile_max_x_ov, tile_max_y_ov)
            tiles.append({'interior': interior_bbox, 'expanded': expanded_bbox})

    client = Client()
    all_points_future = client.scatter(all_points, broadcast=True)
    print("Triangulating each tile in parallel...")
    futures = [client.submit(process_tile_dask, t, all_points_future) for t in tiles]
    results = []
    for fut in tqdm(futures, desc="Tiling", unit="tile"):
        results.append(fut.result())
    client.close()

    print("Merging tiled sub-meshes...")
    sub_meshes = []
    for res in results:
        if res is not None:
            verts, faces = res
            sub_meshes.append(trimesh.Trimesh(vertices=verts, faces=faces))
    if not sub_meshes:
        messagebox.showerror("Error", "No valid triangulations. Exiting.")
        return

    merged_mesh = trimesh.util.concatenate(sub_meshes)
    merged_mesh.export(output_stl, file_type='stl')
    print(f"Successfully exported {output_stl}")

####################################
# Minimal STL -> LandXML Converter
####################################
def convert_stl_to_landxml(input_stl: str, output_xml: str) -> None:
    """
    Hard-coded minimal LandXML export: no user-specified project/app info.
    """
    print(f"Converting STL to LandXML: {input_stl} -> {output_xml}")
    stl_mesh = mesh.Mesh.from_file(input_stl)
    all_points = stl_mesh.vectors.reshape(-1, 3)
    points = np.round(all_points, 4)
    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
    # If you need to swap X/Y for your workflow, uncomment:
    # unique_points = unique_points[:, [1, 0, 2]]
    faces = inverse.reshape(-1, 3)

    points_block_lines = [
        f'    <P id="{i+1}"> {pt[0]} {pt[1]} {pt[2]} </P>' for i, pt in enumerate(unique_points)
    ]
    points_block = "\n".join(points_block_lines)
    faces_block_lines = [
        f'    <F> {face[0]+1} {face[1]+1} {face[2]+1} </F>' for face in faces
    ]
    faces_block = "\n".join(faces_block_lines)

    surface_name = os.path.basename(output_xml)
    xml_content = LANDXML_TEMPLATE.format(
        surface_name=surface_name,
        points_block=points_block,
        faces_block=faces_block
    )
    with open(output_xml, 'w') as f:
        f.write(xml_content)
    print(f"LandXML written to: {output_xml}")

####################################
# STL Viewer using Open3D (integrated from stl_viewer.py)
####################################

def view_stl(stl_file: str):
    if o3d is None:
        messagebox.showerror("Open3D Error", "Open3D is not installed. Cannot view STL.")
        return

    # Create a larger loading window
    loading_win = tk.Toplevel()
    loading_win.title("Loading Mesh")
    loading_win.geometry("400x200")  # Increase the window size
    label = tk.Label(loading_win, text="Loading mesh, please wait...", font=("Helvetica", 14))
    label.pack(padx=20, pady=20)
    progress_bar = ttk.Progressbar(loading_win, mode="indeterminate", length=300)
    progress_bar.pack(padx=20, pady=20)
    progress_bar.start(10)  # Animation speed
    loading_win.update()

    # Load the mesh (this may take some time)
    mesh_o3d = o3d.io.read_triangle_mesh(stl_file)
    
    # Once loaded, stop and close the loading window
    progress_bar.stop()
    loading_win.destroy()

    if mesh_o3d.is_empty():
        messagebox.showerror("Error", "Mesh is empty or failed to load.")
        return

    # Compute normals and center the mesh
    mesh_o3d.compute_vertex_normals()
    center = mesh_o3d.get_center()
    mesh_o3d.translate(-center)
    
    # Create an Open3D visualizer window and adjust rendering options
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Fast STL Viewer", width=1280, height=720)
    opt = vis.get_render_option()
    opt.mesh_show_wireframe = False  # Disable wireframe lines so the mesh appears smooth
    vis.add_geometry(mesh_o3d)
    vis.run()
    vis.destroy_window()

####################################
# UI Implementation
####################################
def run_conversion():
    input_las = entry_input.get()
    output_stl = entry_output.get()

    # Parse or fall back to defaults if empty:
    def floatval_or_default(entry_widget, default):
        val_str = entry_widget.get().strip()
        if not val_str:
            return default
        try:
            return float(val_str)
        except ValueError:
            return default

    def intval_or_default(entry_widget, default):
        val_str = entry_widget.get().strip()
        if not val_str:
            return default
        try:
            return int(val_str)
        except ValueError:
            return default

    # Gather decimation parameters, using defaults if the field is empty or invalid
    adaptive_grid = intval_or_default(entry_adaptive_grid, 1000)
    adaptive_minfrac = floatval_or_default(entry_adaptive_minfrac, 0.0005)
    adaptive_maxfrac = floatval_or_default(entry_adaptive_maxfrac, 0.1)
    curve_exponent = floatval_or_default(entry_curve_exponent, 0.4)
    flat_threshold = floatval_or_default(entry_flat_threshold, 0.008)
    flat_fraction = floatval_or_default(entry_flat_fraction, 0.0)
    curb_threshold = floatval_or_default(entry_curb_threshold, 0.065)

    use_gpu = use_gpu_decimation_var.get()
    export_xyz_path = entry_xyz_output.get() if export_xyz_var.get() else None

    try:
        convert_las_to_stl(
            input_las, output_stl,
            adaptive_grid=adaptive_grid,
            adaptive_minfrac=adaptive_minfrac,
            adaptive_maxfrac=adaptive_maxfrac,
            curve_exponent=curve_exponent,
            flat_threshold=flat_threshold,
            flat_fraction=flat_fraction,
            curb_threshold=curb_threshold,
            use_gpu_decimation=use_gpu,
            output_xyz=export_xyz_path
        )
    except Exception as e:
        messagebox.showerror("Error", f"LAS -> STL failed: {e}")
        return

    # Minimal LandXML export if selected
    if export_landxml_var.get():
        output_landxml = entry_landxml_output.get()
        if not output_landxml:
            messagebox.showerror("Error", "Please specify an output LandXML file.")
            return
        try:
            convert_stl_to_landxml(output_stl, output_landxml)
            messagebox.showinfo("Success", f"Exported STL + LandXML:\n{output_stl}\n{output_landxml}")
        except Exception as e:
            messagebox.showerror("Error", f"STL -> LandXML failed: {e}")
    else:
        messagebox.showinfo("Success", f"Exported STL: {output_stl}")

    # Prompt user to view the STL using the integrated viewer
    if messagebox.askyesno("View STL", "Do you want to view the generated STL file?"):
        view_stl(output_stl)

def toggle_landxml_frame():
    if export_landxml_var.get():
        landxml_frame.grid(row=landxml_row, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
    else:
        landxml_frame.grid_remove()

def toggle_xyz_frame():
    if export_xyz_var.get():
        xyz_frame.grid(row=xyz_row, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
    else:
        xyz_frame.grid_remove()

def browse_input_file():
    filename = filedialog.askopenfilename(filetypes=[("LAS Files", "*.las")])
    if filename:
        entry_input.delete(0, tk.END)
        entry_input.insert(0, filename)

def browse_output_file():
    filename = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL Files", "*.stl")])
    if filename:
        entry_output.delete(0, tk.END)
        entry_output.insert(0, filename)

def browse_landxml_file():
    filename = filedialog.asksaveasfilename(defaultextension=".xml", filetypes=[("XML Files", "*.xml")])
    if filename:
        entry_landxml_output.delete(0, tk.END)
        entry_landxml_output.insert(0, filename)

def browse_xyz_file():
    filename = filedialog.asksaveasfilename(defaultextension=".las", filetypes=[("LAS Files", "*.las")])
    if filename:
        entry_xyz_output.delete(0, tk.END)
        entry_xyz_output.insert(0, filename)

####################################
# Main UI
####################################
def main():
    global entry_input, entry_output
    global entry_adaptive_grid, entry_adaptive_minfrac, entry_adaptive_maxfrac
    global entry_curve_exponent, entry_flat_threshold, entry_flat_fraction, entry_curb_threshold
    global use_gpu_decimation_var, export_xyz_var, xyz_frame, entry_xyz_output, xyz_row
    global export_landxml_var, landxml_frame, entry_landxml_output, landxml_row

    root = tk.Tk()
    root.title("LAS -> STL -> (optional) LandXML / LAS Converter")
    main_frame = tk.Frame(root, padx=10, pady=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    row_idx = 0

    # LAS input / STL output
    tk.Label(main_frame, text="Input LAS File:").grid(row=row_idx, column=0, sticky=tk.W)
    entry_input = tk.Entry(main_frame, width=50)
    entry_input.grid(row=row_idx, column=1)
    tk.Button(main_frame, text="Browse", command=browse_input_file).grid(row=row_idx, column=2)
    row_idx += 1

    tk.Label(main_frame, text="Output STL File:").grid(row=row_idx, column=0, sticky=tk.W)
    entry_output = tk.Entry(main_frame, width=50)
    entry_output.grid(row=row_idx, column=1)
    tk.Button(main_frame, text="Browse", command=browse_output_file).grid(row=row_idx, column=2)
    row_idx += 1

    # Decimation parameters (with tooltips)
    adaptive_label = tk.Label(main_frame, text="Adaptive Decimation Grid (opt):")
    adaptive_label.grid(row=row_idx, column=0, sticky=tk.W)
    CreateToolTip(adaptive_label, "AAdaptive Decimation Grid (set to 1000 for starting point): This is the resolution at which the decimation occurs. The script will split the entire model up into a grid in this resolution. In this case 1000x1000. This is used to detect where the curbs are and keeps the density in the cells that curbs are detected. Additionally, it keeps the nearby cells dense as a buffer. The larger it is the more fine the resolution, but also the risk of blowing out the triangles.")
    entry_adaptive_grid = tk.Entry(main_frame, width=50)
    entry_adaptive_grid.insert(0, "1000")
    entry_adaptive_grid.grid(row=row_idx, column=1)
    row_idx += 1

    minfrac_label = tk.Label(main_frame, text="Min Fraction (opt):")
    minfrac_label.grid(row=row_idx, column=0, sticky=tk.W)
    CreateToolTip(minfrac_label, "Min Fraction (set to 0.0005 for starting point): This is the amount of triangles that will remain in a grid cell at the most flat area. This does not include the areas that are considered perfectly flat, which is governed by the flat fraction instead.")
    entry_adaptive_minfrac = tk.Entry(main_frame, width=50)
    entry_adaptive_minfrac.insert(0, "0.0005")
    entry_adaptive_minfrac.grid(row=row_idx, column=1)
    row_idx += 1

    maxfrac_label = tk.Label(main_frame, text="Max Fraction (opt):")
    maxfrac_label.grid(row=row_idx, column=0, sticky=tk.W)
    CreateToolTip(maxfrac_label, "Max Fraction (set to 0.1 for starting point): This is the amount of triangles that will remain in a grid cell at the least flat area. This does not include the curb cells, which is hard-coded to keep 100 percent of the points.")
    entry_adaptive_maxfrac = tk.Entry(main_frame, width=50)
    entry_adaptive_maxfrac.insert(0, "0.1")
    entry_adaptive_maxfrac.grid(row=row_idx, column=1)
    row_idx += 1

    curve_label = tk.Label(main_frame, text="Curve Exponent (opt):")
    curve_label.grid(row=row_idx, column=0, sticky=tk.W)
    CreateToolTip(curve_label, "Curve Exponent (set to 0.4 for starting point): This is the mathematical curve between the min and max fractions. Basically just sets how much the decimation ramps up to the max fraction. Higher values favor the min fraction and produces less triangles.")
    entry_curve_exponent = tk.Entry(main_frame, width=50)
    entry_curve_exponent.insert(0, "0.4")
    entry_curve_exponent.grid(row=row_idx, column=1)
    row_idx += 1

    flat_thresh_label = tk.Label(main_frame, text="Flat Threshold (opt):")
    flat_thresh_label.grid(row=row_idx, column=0, sticky=tk.W)
    CreateToolTip(flat_thresh_label, "Flat Threshold (set to 0.008 for starting point): This effects the amount of ground that will be considered completely flat. The higher the value the more areas that will be considered completely flat.")
    entry_flat_threshold = tk.Entry(main_frame, width=50)
    entry_flat_threshold.insert(0, "0.008")
    entry_flat_threshold.grid(row=row_idx, column=1)
    row_idx += 1

    flat_frac_label = tk.Label(main_frame, text="Flat Fraction (opt):")
    flat_frac_label.grid(row=row_idx, column=0, sticky=tk.W)
    CreateToolTip(flat_frac_label, "Flat Fraction (set to 0 for starting point): This is the minimum amount of triangles that must remain in a cell when it is considered completely flat. Set only to very low values such as 0 or 0.000001.")
    entry_flat_fraction = tk.Entry(main_frame, width=50)
    entry_flat_fraction.insert(0, "0")
    entry_flat_fraction.grid(row=row_idx, column=1)
    row_idx += 1

    curb_label = tk.Label(main_frame, text="Curb Edge Threshold (m, opt):")
    curb_label.grid(row=row_idx, column=0, sticky=tk.W)
    CreateToolTip(curb_label, "Curb Edge Threshold (set to 0.065 for starting point): This is the distance that will calculate whether or not an area is a curb. The higher the values the higher an cliff needs to be to be considered a curb.")
    entry_curb_threshold = tk.Entry(main_frame, width=50)
    entry_curb_threshold.insert(0, "0.065")
    entry_curb_threshold.grid(row=row_idx, column=1)
    row_idx += 1

    # GPU decimation checkbox
    use_gpu_decimation_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        main_frame,
        text="Use GPU for Decimation?",
        variable=use_gpu_decimation_var
    ).grid(row=row_idx, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
    row_idx += 1

    # Optional export LAS
    export_xyz_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        main_frame,
        text="Export LAS Point Cloud?",
        variable=export_xyz_var,
        command=toggle_xyz_frame
    ).grid(row=row_idx, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
    row_idx += 1
    xyz_row = row_idx
    row_idx += 1

    xyz_frame = tk.Frame(main_frame, borderwidth=1, relief=tk.RIDGE, padx=5, pady=5)
    tk.Label(xyz_frame, text="Output LAS File:").grid(row=0, column=0, sticky=tk.W)
    entry_xyz_output = tk.Entry(xyz_frame, width=50)
    entry_xyz_output.grid(row=0, column=1)
    tk.Button(xyz_frame, text="Browse", command=browse_xyz_file).grid(row=0, column=2)
    xyz_frame.grid(row=xyz_row, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
    xyz_frame.grid_remove()

    # Optional export LandXML (minimal)
    export_landxml_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        main_frame,
        text="Export LandXML?",
        variable=export_landxml_var,
        command=toggle_landxml_frame
    ).grid(row=row_idx, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
    row_idx += 1
    landxml_row = row_idx
    row_idx += 1

    landxml_frame = tk.Frame(main_frame, borderwidth=1, relief=tk.RIDGE, padx=5, pady=5)
    tk.Label(landxml_frame, text="Output LandXML File:").grid(row=0, column=0, sticky=tk.W)
    entry_landxml_output = tk.Entry(landxml_frame, width=50)
    entry_landxml_output.grid(row=0, column=1)
    tk.Button(landxml_frame, text="Browse", command=browse_landxml_file).grid(row=0, column=2)
    landxml_frame.grid(row=landxml_row, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
    landxml_frame.grid_remove()

    # Convert button
    tk.Button(
        main_frame,
        text="Convert",
        command=run_conversion
    ).grid(row=row_idx, column=0, columnspan=3, pady=(10, 0))

    root.mainloop()

if __name__ == "__main__":
    main()

