# LAS to STL Converter

A high-performance Python tool for converting LiDAR point cloud data (LAS/LAZ) to 3D mesh models (STL) with intelligent adaptive decimation and curb detection.

## Features

- Convert LAS files to STL mesh models
- GPU-accelerated adaptive decimation (CUDA support)
- Curb detection and preservation
- Optional LandXML export
- Built-in STL viewer using Open3D
- Multi-threaded processing using Dask
- GUI interface

## Requirements

```
dask==2025.1.0
distributed==2025.1.0
laspy==2.5.4
numpy==2.2.3
numpy_stl==3.2.0
open3d==0.19.0
pymeshlab==2023.12.post2
scipy==1.15.2
tqdm==4.66.5
trimesh==4.6.0
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Compile the decimation libraries:

### Linux
CPU support:
```bash
gcc -shared -o libdecimation.so decimation.c -fPIC -lm
```

GPU support:
```bash
nvcc -shared -o libdecimation_cuda.so decimation_cuda.cu -Xcompiler -fPIC
```

### Windows
CPU support:
```bash
gcc -shared -o decimation.dll decimation.c
```

GPU support:
```bash
nvcc -shared -o decimation_cuda.dll decimation_cuda.cu
```

Note: For Windows compilation, ensure you have MinGW-w64 for gcc and CUDA toolkit installed for nvcc.

## Usage

Launch the GUI:
```bash
python combined_script.py
```

Or use the viewer directly:
```bash
python stl_viewer.py your_model.stl
```

## Decimation Parameters

- **Adaptive Grid** (set to 1000 for starting point): This is the resolution at which the decimation occurs. The script will split the entire model up into a grid in this resolution. In this case 1000x1000. This is used to detect where the curbs are and keeps the density in the cells that curbs are detected. Additionally, it keeps the nearby cells dense as a buffer. The larger it is the more fine the resolution, but also the risk of blowing out the triangles.

- **Min Fraction** (set to 0.0005 for starting point): This is the amount of triangles that will remain in a grid cell at the most flat area. This does not include the areas that are considered perfectly flat, which is governed by the flat fraction instead.

- **Max Fraction** (set to 0.1 for starting point): This is the amount of triangles that will remain in a grid cell at the least flat area. This does not include the curb cells, which is hard-coded to keep 100 percent of the points.

- **Curve Exponent** (set to 0.4 for starting point): This is the mathematical curve between the min and max fractions. Basically just sets how much the decimation ramps up to the max fraction. Higher values favor the min fraction and produces less triangles.

- **Flat Threshold** (set to 0.008 for starting point): This effects the amount of ground that will be considered completely flat. The higher the value the more areas that will be considered completely flat.

- **Flat Fraction** (set to 0 for starting point): This is the minimum amount of triangles that must remain in a cell when it is considered completely flat. Set only to very low values such as 0 or 0.000001.

- **Curb Edge Threshold** (set to 0.065 for starting point): This is the distance that will calculate whether or not an area is a curb. The higher the values the higher a cliff needs to be to be considered a curb.

## Features in Detail

### Adaptive Decimation
The tool uses an intelligent adaptive decimation algorithm that:
- Preserves detail in areas of high complexity
- Reduces point density in flat regions
- Maintains curb features with high fidelity
- Supports both CPU and GPU processing

### Curb Detection
Employs a sophisticated algorithm to:
- Detect vertical discontinuities in the point cloud
- Preserve full detail around curb features
- Maintain higher point density in adjacent areas

### Multi-format Support
- Input: LAS/LAZ point cloud files
- Output: STL mesh files
- Optional: LandXML export for CAD compatibility
- Optional: Decimated point cloud export

## Performance

The tool uses several optimization techniques:
- Parallel processing with Dask
- CUDA-accelerated decimation (when available)
- Efficient memory management for large datasets
- Tiled processing for handling large point clouds

## Acknowledgments

Built with:
- Open3D for visualization
- CUDA for GPU acceleration
- Dask for distributed computing
- laspy for LAS/LAZ handling