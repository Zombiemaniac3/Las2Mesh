#!/usr/bin/env python3 
"""
LAS to STL and optional STL to LandXML/PCD Converter with a dynamic UI

This script:
  - Converts a LAS file to an STL mesh.
  - Optionally converts that STL mesh to a LandXML file containing a TIN surface.
  - Optionally exports the (decimated) point cloud as a LAS file.
  
The UI includes:
  - Fields for LAS → STL conversion.
  - Checkboxes to enable/disable LandXML export, LAS export, and GPU decimation.

Dependencies:
  - laspy
  - numpy
  - scipy
  - trimesh
  - tqdm
  - tkinter (built-in)
  - numpy-stl
  - dask[distributed]
  
Optional:
  - pymeshlab (for advanced mesh cleaning)
  
Usage:
  python combined_script.py
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

# Use dask for distributed parallelism (across nodes/cores)
from dask.distributed import Client

try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False

####################################
# New LandXML template
####################################
LANDXML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<LandXML version="1.2" xmlns="http://www.landxml.org/schema/LandXML-1.2">
  <Application name="{app_name}" desc="{app_desc}" manufacturer="{manufacturer}" version="{app_version}" manufacturerURL="{manufacturerURL}" />
  <Units>
    {units_block}
  </Units>
  <CoordinateSystem name="{cs_full}" epsgCode="{epsg_code}" />
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
# Determine the directory of this script.
script_dir = os.path.dirname(os.path.abspath(__file__))

####################################
# Parallel triangulation utilities using Dask
####################################
def process_tile_dask(tile, all_points):
    """
    Process a tile dictionary containing:
      - 'expanded': expanded bounding box (minx, miny, maxx, maxy)
      - 'interior': interior bounding box (minx, miny, maxx, maxy)
    Filters the provided all_points array to the expanded bbox,
    performs Delaunay triangulation, and clips the resulting mesh to include only
    faces touching the interior bbox.
    """
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
                       output_xyz=None):  # output_xyz now will be treated as a LAS filename.
    # Read the LAS file in chunks.
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

    # Optionally perform adaptive decimation.
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

        # Select which shared library to load based on use_gpu_decimation.
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
            np.ctypeslib.ndpointer(dtype=np.double, flags="C_CONTIGUOUS"),  # input array (n_points*3)
            ctypes.c_int,    # n_points
            ctypes.c_int,    # grid_cells
            ctypes.c_double, # min_fraction
            ctypes.c_double, # max_fraction
            ctypes.c_double, # curve_exponent
            ctypes.c_double, # flat_threshold
            ctypes.c_double, # flat_fraction
            ctypes.c_double, # curb_edge_threshold
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # out_points (pointer-to-pointer)
            ctypes.POINTER(ctypes.c_int)  # out_n_points
        ]
        func.restype = ctypes.c_int

        # Prepare output variables.
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
        # Create a NumPy array from the C output.
        decimated_points = np.ctypeslib.as_array(out_points, shape=(kept_count, 3)).copy()
        # Free the memory allocated in C.
        if os.name != "nt":
            libc = ctypes.CDLL("libc.so.6")
            libc.free.argtypes = [ctypes.c_void_p]
            libc.free(ctypes.cast(out_points, ctypes.c_void_p))
        else:
            # On Windows, skipping freeing to avoid CRT heap issues.
            pass
        after_count = decimated_points.shape[0]
        print(f"Retained {after_count} / {before_count} points after adaptive decimation.")
        all_points = decimated_points

    # Optional: Export the (decimated) point cloud as a LAS file.
    if output_xyz is not None and output_xyz.strip():
        # Create a new LAS header and file.
        # Here we set the scale factors and offsets based on the min values.
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.x_offset = np.min(all_points[:, 0])
        header.y_offset = np.min(all_points[:, 1])
        header.z_offset = np.min(all_points[:, 2])
        # Set scale factors (adjust these if needed).
        header.x_scale = 0.001
        header.y_scale = 0.001
        header.z_scale = 0.001

        las = laspy.LasData(header)
        las.x = all_points[:, 0]
        las.y = all_points[:, 1]
        las.z = all_points[:, 2]
        las.write(output_xyz)
        print(f"Successfully exported decimated point cloud to LAS: {output_xyz}")

    # Determine the bounding box.
    min_x, min_y = np.min(all_points[:, :2], axis=0)
    max_x, max_y = np.max(all_points[:, :2], axis=0)

    # Set up tiling parameters.
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

    # Set up a Dask client (which can span multiple nodes if configured appropriately).
    client = Client()
    # Scatter all_points so that every worker has it.
    all_points_future = client.scatter(all_points, broadcast=True)
    print("Triangulating each tile in parallel across the cluster...")
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
    print(" NOT Cleaning up duplicates and degenerates...")
    #merged_mesh.merge_vertices()
    #merged_mesh.remove_duplicate_faces()
    #merged_mesh.remove_degenerate_faces()
    print("Exporting final mesh to STL...")
    merged_mesh.export(output_stl, file_type='stl')
    print(f"Successfully exported {output_stl}")

####################################
# STL -> LandXML converter (Optimized and Updated)
####################################
def convert_stl_to_landxml(input_stl: str, output_xml: str,
                           project_name: str, project_desc: str,
                           app_name: str, manufacturer: str, app_version: str,
                           cs_name: str, cs_datum: str, cs_zone: str,
                           units_linear: str, units_area: str, units_volume: str) -> None:
    print(f"Converting STL to LandXML: {input_stl} -> {output_xml}")
    stl_mesh = mesh.Mesh.from_file(input_stl)
    all_points = stl_mesh.vectors.reshape(-1, 3)
    points = np.round(all_points, 4)
    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
    # Swap x and y columns so that LandXML receives them in the proper order.
    unique_points = unique_points[:, [1, 0, 2]]
    faces = inverse.reshape(-1, 3)
    points_block_lines = [
        f'    <P id="{i+1}"> {pt[0]} {pt[1]} {pt[2]} </P>' for i, pt in enumerate(unique_points)
    ]
    points_block = "\n".join(points_block_lines)
    faces_block_lines = [
        f'    <F> {face[0]+1} {face[1]+1} {face[2]+1} </F>' for face in faces
    ]
    faces_block = "\n".join(faces_block_lines)
    app_desc = "LandXML Export"
    manufacturerURL = "http://www.example.com"
    # Simple handling for units:
    if units_linear.lower() == "meter":
        units_block = '<Metric linearUnit="meter" elevationUnit="meter" />'
    elif units_linear.lower() in ["international foot", "us survey foot"]:
        units_block = '<Imperial linearUnit="foot" elevationUnit="feet" />'
    else:
        units_block = '<Imperial linearUnit="foot" elevationUnit="feet" />'
    # Construct coordinate system string.
    if cs_name == "StatePlane":
        unit_desc = units_linear.lower()
        cs_full = f"State Plane ({cs_zone}) / {cs_datum} / {unit_desc}"
        # Example EPSG code logic.
        if cs_zone == "Arizona Central" and cs_datum in ["NAD83(2011)", "NAD83"]:
            epsg_code = "6405" if cs_datum == "NAD83(2011)" else "2223"
        else:
            epsg_code = "0000"
    else:
        cs_full = cs_name
        epsg_code = "0000"
    surface_name = os.path.basename(output_xml)
    xml_content = LANDXML_TEMPLATE.format(
        app_name=app_name,
        app_desc=app_desc,
        manufacturer=manufacturer,
        app_version=app_version,
        manufacturerURL=manufacturerURL,
        units_block=units_block,
        cs_full=cs_full,
        epsg_code=epsg_code,
        surface_name=surface_name,
        points_block=points_block,
        faces_block=faces_block,
        project_name=project_name,
        project_desc=project_desc
    )
    with open(output_xml, 'w') as f:
        f.write(xml_content)
    print(f"LandXML written to: {output_xml}")

####################################
# UI Implementation
####################################
def run_conversion():
    input_las = entry_input.get()
    output_stl = entry_output.get()
    try:
        adaptive_grid = int(entry_adaptive_grid.get()) if entry_adaptive_grid.get() else None
    except ValueError:
        messagebox.showerror("Error", "Adaptive Grid must be an integer.")
        return
    try:
        adaptive_minfrac = float(entry_adaptive_minfrac.get()) if entry_adaptive_minfrac.get() else 0.1
    except ValueError:
        messagebox.showerror("Error", "Min Fraction must be float.")
        return
    try:
        adaptive_maxfrac = float(entry_adaptive_maxfrac.get()) if entry_adaptive_maxfrac.get() else 1.0
    except ValueError:
        messagebox.showerror("Error", "Max Fraction must be float.")
        return
    try:
        curve_exponent = float(entry_curve_exponent.get()) if entry_curve_exponent.get() else 1.0
    except ValueError:
        messagebox.showerror("Error", "Curve Exponent must be float.")
        return
    try:
        flat_threshold = float(entry_flat_threshold.get()) if entry_flat_threshold.get() else 0.0
    except ValueError:
        messagebox.showerror("Error", "Flat Threshold must be float.")
        return
    try:
        flat_fraction = float(entry_flat_fraction.get()) if entry_flat_fraction.get() else 0.0
    except ValueError:
        messagebox.showerror("Error", "Flat Fraction must be float.")
        return
    try:
        curb_threshold = float(entry_curb_threshold.get()) if entry_curb_threshold.get() else 0.2
    except ValueError:
        messagebox.showerror("Error", "Curb Edge Threshold must be float.")
        return

    use_gpu = use_gpu_decimation_var.get()
    # Get optional LAS export filename (if specified)
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

    if export_landxml_var.get():
        output_landxml = entry_landxml_output.get()
        if not output_landxml:
            messagebox.showerror("Error", "Please specify an output LandXML file.")
            return
        project_name = entry_lxml_project_name.get()
        project_desc = entry_lxml_project_desc.get()
        app_name = entry_lxml_app_name.get()
        manufacturer = entry_lxml_manufacturer.get()
        app_version = entry_lxml_app_version.get()
        cs_name = var_cs_name.get()
        cs_datum = var_cs_datum.get()
        cs_zone = var_cs_zone.get()
        units_linear = var_units_linear.get()
        units_area = var_units_area.get()
        units_volume = var_units_volume.get()
        try:
            convert_stl_to_landxml(
                input_stl=output_stl,
                output_xml=entry_landxml_output.get(),
                project_name=project_name,
                project_desc=project_desc,
                app_name=app_name,
                manufacturer=manufacturer,
                app_version=app_version,
                cs_name=cs_name,
                cs_datum=cs_datum,
                cs_zone=cs_zone,
                units_linear=units_linear,
                units_area=units_area,
                units_volume=units_volume
            )
            messagebox.showinfo("Success", f"Exported STL + LandXML:\n{output_stl}\n{entry_landxml_output.get()}")
        except Exception as e:
            messagebox.showerror("Error", f"STL -> LandXML failed: {e}")
    else:
        messagebox.showinfo("Success", f"Exported STL: {output_stl}")

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
    # Updated to default to .las instead of .pcd.
    filename = filedialog.asksaveasfilename(defaultextension=".las", filetypes=[("LAS Files", "*.las")])
    if filename:
        entry_xyz_output.delete(0, tk.END)
        entry_xyz_output.insert(0, filename)

####################################
# Start of main UI
####################################
def main():
    global entry_input, entry_output
    global entry_adaptive_grid, entry_adaptive_minfrac, entry_adaptive_maxfrac
    global entry_curve_exponent, entry_flat_threshold, entry_flat_fraction, entry_curb_threshold
    global export_landxml_var, landxml_frame, entry_landxml_output, landxml_row
    global entry_lxml_project_name, entry_lxml_project_desc
    global entry_lxml_app_name, entry_lxml_manufacturer, entry_lxml_app_version
    global var_cs_name, var_cs_datum, var_cs_zone
    global var_units_linear, var_units_area, var_units_volume
    global use_gpu_decimation_var, export_xyz_var, xyz_frame, entry_xyz_output, xyz_row

    root = tk.Tk()
    root.title("LAS -> STL -> (optional) LandXML / LAS Converter")
    main_frame = tk.Frame(root, padx=10, pady=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    row_idx = 0

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

    tk.Label(main_frame, text="Adaptive Decimation Grid (opt):").grid(row=row_idx, column=0, sticky=tk.W)
    entry_adaptive_grid = tk.Entry(main_frame, width=50)
    entry_adaptive_grid.grid(row=row_idx, column=1)
    row_idx += 1

    tk.Label(main_frame, text="Min Fraction (opt):").grid(row=row_idx, column=0, sticky=tk.W)
    entry_adaptive_minfrac = tk.Entry(main_frame, width=50)
    entry_adaptive_minfrac.grid(row=row_idx, column=1)
    row_idx += 1

    tk.Label(main_frame, text="Max Fraction (opt):").grid(row=row_idx, column=0, sticky=tk.W)
    entry_adaptive_maxfrac = tk.Entry(main_frame, width=50)
    entry_adaptive_maxfrac.grid(row=row_idx, column=1)
    row_idx += 1

    tk.Label(main_frame, text="Curve Exponent (opt):").grid(row=row_idx, column=0, sticky=tk.W)
    entry_curve_exponent = tk.Entry(main_frame, width=50)
    entry_curve_exponent.grid(row=row_idx, column=1)
    row_idx += 1

    tk.Label(main_frame, text="Flat Threshold (opt):").grid(row=row_idx, column=0, sticky=tk.W)
    entry_flat_threshold = tk.Entry(main_frame, width=50)
    entry_flat_threshold.grid(row=row_idx, column=1)
    row_idx += 1

    tk.Label(main_frame, text="Flat Fraction (opt):").grid(row=row_idx, column=0, sticky=tk.W)
    entry_flat_fraction = tk.Entry(main_frame, width=50)
    entry_flat_fraction.grid(row=row_idx, column=1)
    row_idx += 1

    tk.Label(main_frame, text="Curb Edge Threshold (m, opt):").grid(row=row_idx, column=0, sticky=tk.W)
    entry_curb_threshold = tk.Entry(main_frame, width=50)
    entry_curb_threshold.grid(row=row_idx, column=1)
    row_idx += 1

    # Add a checkbox to select GPU decimation.
    use_gpu_decimation_var = tk.BooleanVar(value=False)
    tk.Checkbutton(main_frame, text="Use GPU for Decimation?", variable=use_gpu_decimation_var).grid(
        row=row_idx, column=0, columnspan=3, sticky=tk.W, pady=(10, 0)
    )
    row_idx += 1

    # Checkbox for Export LAS and its file selection.
    export_xyz_var = tk.BooleanVar(value=False)
    tk.Checkbutton(main_frame, text="Export LAS Point Cloud?", variable=export_xyz_var, command=toggle_xyz_frame).grid(
        row=row_idx, column=0, columnspan=3, sticky=tk.W, pady=(10, 0)
    )
    row_idx += 1
    xyz_row = row_idx
    xyz_frame = tk.Frame(main_frame, borderwidth=1, relief=tk.RIDGE, padx=5, pady=5)
    tk.Label(xyz_frame, text="Output LAS File:").grid(row=0, column=0, sticky=tk.W)
    entry_xyz_output = tk.Entry(xyz_frame, width=50)
    entry_xyz_output.grid(row=0, column=1)
    tk.Button(xyz_frame, text="Browse", command=browse_xyz_file).grid(row=0, column=2)
    xyz_frame.grid(row=xyz_row, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
    xyz_frame.grid_remove()  # hide by default

    export_landxml_var = tk.BooleanVar(value=False)
    tk.Checkbutton(main_frame, text="Export LandXML?", variable=export_landxml_var, command=toggle_landxml_frame).grid(
        row=row_idx, column=0, columnspan=3, sticky=tk.W, pady=(10, 0)
    )
    row_idx += 1

    landxml_row = row_idx
    landxml_frame = tk.Frame(main_frame, borderwidth=1, relief=tk.RIDGE, padx=5, pady=5)
    tk.Label(landxml_frame, text="Output LandXML File:").grid(row=0, column=0, sticky=tk.W)
    entry_landxml_output = tk.Entry(landxml_frame, width=50)
    entry_landxml_output.grid(row=0, column=1)
    tk.Button(landxml_frame, text="Browse", command=browse_landxml_file).grid(row=0, column=2)
    tk.Label(landxml_frame, text="Project Name:").grid(row=1, column=0, sticky=tk.W)
    entry_lxml_project_name = tk.Entry(landxml_frame, width=50)
    entry_lxml_project_name.grid(row=1, column=1)
    tk.Label(landxml_frame, text="Project Description:").grid(row=2, column=0, sticky=tk.W)
    entry_lxml_project_desc = tk.Entry(landxml_frame, width=50)
    entry_lxml_project_desc.grid(row=2, column=1)
    tk.Label(landxml_frame, text="Application Name:").grid(row=3, column=0, sticky=tk.W)
    entry_lxml_app_name = tk.Entry(landxml_frame, width=50)
    entry_lxml_app_name.grid(row=3, column=1)
    tk.Label(landxml_frame, text="Manufacturer:").grid(row=4, column=0, sticky=tk.W)
    entry_lxml_manufacturer = tk.Entry(landxml_frame, width=50)
    entry_lxml_manufacturer.grid(row=4, column=1)
    tk.Label(landxml_frame, text="Application Version:").grid(row=5, column=0, sticky=tk.W)
    entry_lxml_app_version = tk.Entry(landxml_frame, width=50)
    entry_lxml_app_version.grid(row=5, column=1)
    cs_options = ["StatePlane", "UTM", "WGS84", "Custom"]
    datum_options = ["NAD83(2011)", "NAD83", "NAD27", "WGS84", "ETRS89"]
    zone_options = ["Arizona Central", "0000", "0601", "Custom"]
    tk.Label(landxml_frame, text="Coord System Name:").grid(row=6, column=0, sticky=tk.W)
    var_cs_name = tk.StringVar(value=cs_options[0])
    cs_menu = tk.OptionMenu(landxml_frame, var_cs_name, *cs_options)
    cs_menu.config(width=30)
    cs_menu.grid(row=6, column=1, sticky=tk.W)
    tk.Label(landxml_frame, text="Coord System Datum:").grid(row=7, column=0, sticky=tk.W)
    var_cs_datum = tk.StringVar(value=datum_options[0])
    datum_menu = tk.OptionMenu(landxml_frame, var_cs_datum, *datum_options)
    datum_menu.config(width=30)
    datum_menu.grid(row=7, column=1, sticky=tk.W)
    tk.Label(landxml_frame, text="Coord System Zone:").grid(row=8, column=0, sticky=tk.W)
    var_cs_zone = tk.StringVar(value=zone_options[0])
    zone_menu = tk.OptionMenu(landxml_frame, var_cs_zone, *zone_options)
    zone_menu.config(width=30)
    zone_menu.grid(row=8, column=1, sticky=tk.W)
    linear_units = ["meter", "International Foot", "US Survey Foot"]
    area_units = ["squareMeter", "Square Foot (International)", "Square Foot (US Survey)"]
    volume_units = ["cubicMeter", "Cubic Foot (International)", "Cubic Foot (US Survey)"]
    tk.Label(landxml_frame, text="Linear Unit:").grid(row=9, column=0, sticky=tk.W)
    var_units_linear = tk.StringVar(value=linear_units[0])
    linear_menu = tk.OptionMenu(landxml_frame, var_units_linear, *linear_units)
    linear_menu.config(width=30)
    linear_menu.grid(row=9, column=1, sticky=tk.W)
    tk.Label(landxml_frame, text="Area Unit:").grid(row=10, column=0, sticky=tk.W)
    var_units_area = tk.StringVar(value=area_units[0])
    area_menu = tk.OptionMenu(landxml_frame, var_units_area, *area_units)
    area_menu.config(width=30)
    area_menu.grid(row=10, column=1, sticky=tk.W)
    tk.Label(landxml_frame, text="Volume Unit:").grid(row=11, column=0, sticky=tk.W)
    var_units_volume = tk.StringVar(value=volume_units[0])
    volume_menu = tk.OptionMenu(landxml_frame, var_units_volume, *volume_units)
    volume_menu.config(width=30)
    volume_menu.grid(row=11, column=1, sticky=tk.W)
    landxml_frame.grid(row=landxml_row, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
    landxml_frame.grid_remove()
    row_idx = landxml_row + 1

    tk.Button(main_frame, text="Convert", command=run_conversion).grid(row=row_idx, column=0, columnspan=3, pady=(10, 0))
    root.mainloop()

if __name__ == "__main__":
    main()
