// decimation_cuda.cu

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

// ----- User-defined parameters for curb detection and local storage.
#define MAX_POINTS_PER_CELL 4096
#define SUBDIV_FACTOR 4
#define MIN_POINTS_FOR_CURB 12
#define MIN_UPPER_POINTS 8
#define CURB_PERCENT 1

// =======================================================================
// A simple kernel to reorder points from an interleaved array using an index array.
__global__
void reorder_kernel(const double* points, const int* indices, double* sorted_points, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        int src = indices[i];
        sorted_points[3*i]     = points[3*src];
        sorted_points[3*i + 1] = points[3*src + 1];
        sorted_points[3*i + 2] = points[3*src + 2];
    }
}

// =======================================================================
// Kernel to compute the cell ID for each point.
__global__
void compute_cell_ids_kernel(const double* points, int n_points,
                             double min_x, double min_y, double dx, double dy,
                             int grid_cells, int* d_cell_ids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_points) {
        double x = points[3*i];
        double y = points[3*i+1];
        int ix = (int)((x - min_x) / dx);
        int iy = (int)((y - min_y) / dy);
        if(ix < 0) ix = 0;
        if(ix >= grid_cells) ix = grid_cells - 1;
        if(iy < 0) iy = 0;
        if(iy >= grid_cells) iy = grid_cells - 1;
        d_cell_ids[i] = iy * grid_cells + ix;
    }
}

// =======================================================================
// Kernel to decide for each point (in sorted order) whether to keep it.
__global__
void decide_keep_kernel(const int* d_sorted_cell_ids, int n_points,
                        const double* d_cell_fraction, int* d_flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_points) {
        int cid = d_sorted_cell_ids[i];
        double frac = d_cell_fraction[cid];

        // Simple sine-hash pseudo-random.
        double r = sin((double)i * 12.9898) * 43758.5453;
        r = r - floor(r);

        d_flags[i] = (r < frac) ? 1 : 0;
    }
}

// =======================================================================
// Kernel to gather the kept points from the sorted points array.
__global__
void gather_points_kernel(const double* d_points_sorted, const int* d_flags,
                          const int* d_scan, int n_points, double* d_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n_points) {
        if(d_flags[i]) {
            int pos = d_scan[i];
            d_out[3*pos]     = d_points_sorted[3*i];
            d_out[3*pos + 1] = d_points_sorted[3*i + 1];
            d_out[3*pos + 2] = d_points_sorted[3*i + 2];
        }
    }
}

// =======================================================================
// Kernel to process each unique cell segment.
__global__
void per_cell_kernel(const double* d_points_sorted, const int* d_seg_start,
                     const int* d_seg_count, const int* d_unique_cell_ids,
                     int unique_count, double min_x, double min_y, double dx, double dy, int grid_cells,
                     double curb_edge_threshold,
                     double* d_std, int* d_curb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < unique_count) {
        int start = d_seg_start[idx];
        int count = d_seg_count[idx];

        // Compute sums for best-fit plane: sums for x, y, z, xx, xy, yy, xz, yz.
        double sum_x = 0, sum_y = 0, sum_z = 0;
        double sum_xx = 0, sum_xy = 0, sum_yy = 0;
        double sum_xz = 0, sum_yz = 0;
        for (int i = start; i < start + count; i++) {
            double x = d_points_sorted[3*i];
            double y = d_points_sorted[3*i+1];
            double z = d_points_sorted[3*i+2];
            sum_x += x; sum_y += y; sum_z += z;
            sum_xx += x * x;
            sum_xy += x * y;
            sum_yy += y * y;
            sum_xz += x * z;
            sum_yz += y * z;
        }

        // Solve the 3x3 system for best-fit plane coefficients: z ~ a*x + b*y + c.
        double M[3][3] = { { sum_xx, sum_xy, sum_x },
                           { sum_xy, sum_yy, sum_y },
                           { sum_x,  sum_y,  (double)count } };
        double B[3] = { sum_xz, sum_yz, sum_z };
        double coeff[3] = {0, 0, 0};
        bool singular = false;

        // Gaussian elimination in 3x3
        for (int i = 0; i < 3; i++) {
            double max_val = fabs(M[i][i]);
            int pivot = i;
            for (int j = i+1; j < 3; j++) {
                double val = fabs(M[j][i]);
                if(val > max_val) {
                    max_val = val;
                    pivot = j;
                }
            }
            if(max_val < 1e-12) {
                singular = true;
                break;
            }
            if(pivot != i) {
                for (int k = i; k < 3; k++) {
                    double temp = M[i][k];
                    M[i][k] = M[pivot][k];
                    M[pivot][k] = temp;
                }
                double tempB = B[i];
                B[i] = B[pivot];
                B[pivot] = tempB;
            }
            for (int j = i+1; j < 3; j++) {
                double factor = M[j][i] / M[i][i];
                for (int k = i; k < 3; k++) {
                    M[j][k] -= factor * M[i][k];
                }
                B[j] -= factor * B[i];
            }
        }
        if(!singular) {
            for (int i = 2; i >= 0; i--) {
                double sumv = B[i];
                for (int j = i+1; j < 3; j++) {
                    sumv -= M[i][j] * coeff[j];
                }
                coeff[i] = sumv / M[i][i];
            }
        }

        // Compute standard deviation from the plane.
        double sum_res2 = 0;
        for (int i = start; i < start + count; i++) {
            double x = d_points_sorted[3*i];
            double y = d_points_sorted[3*i+1];
            double z = d_points_sorted[3*i+2];
            double z_pred = coeff[0]*x + coeff[1]*y + coeff[2];
            double r = z - z_pred;
            sum_res2 += r*r;
        }
        double std_dev = (count > 0) ? sqrt(sum_res2 / count) : 0.0;
        d_std[idx] = std_dev;

        // --- Curb detection ---
        int cid = d_unique_cell_ids[idx];
        int row = cid / grid_cells;
        int col = cid % grid_cells;
        double cell_x_min = min_x + col * dx;
        double cell_y_min = min_y + row * dy;
        double sub_dx = dx / SUBDIV_FACTOR;
        double sub_dy = dy / SUBDIV_FACTOR;
        int curb_flag = 0;

        for (int sr = 0; sr < SUBDIV_FACTOR && !curb_flag; sr++) {
            for (int sc = 0; sc < SUBDIV_FACTOR && !curb_flag; sc++) {
                double sub_x_min = cell_x_min + sc * sub_dx;
                double sub_y_min = cell_y_min + sr * sub_dy;
                double sub_x_max = sub_x_min + sub_dx;
                double sub_y_max = sub_y_min + sub_dy;

                double z_vals[MAX_POINTS_PER_CELL];
                int sub_count = 0;
                for (int i = start; i < start + count; i++) {
                    double x = d_points_sorted[3*i];
                    double y = d_points_sorted[3*i+1];
                    double z = d_points_sorted[3*i+2];
                    if(x >= sub_x_min && x < sub_x_max &&
                       y >= sub_y_min && y < sub_y_max) {
                        if(sub_count < MAX_POINTS_PER_CELL) {
                            z_vals[sub_count++] = z;
                        }
                    }
                }
                if(sub_count >= MIN_POINTS_FOR_CURB) {
                    // Insertion sort for small sub_count.
                    for (int m = 1; m < sub_count; m++) {
                        double key = z_vals[m];
                        int jj = m - 1;
                        while(jj >= 0 && z_vals[jj] > key) {
                            z_vals[jj+1] = z_vals[jj];
                            jj--;
                        }
                        z_vals[jj+1] = key;
                    }
                    double median = z_vals[sub_count/2];
                    double sum_upper = 0.0;
                    int count_upper = 0;
                    for (int m = 0; m < sub_count; m++) {
                        if(z_vals[m] > median) {
                            sum_upper += z_vals[m];
                            count_upper++;
                        }
                    }
                    if(count_upper >= MIN_UPPER_POINTS) {
                        double mean_upper = sum_upper / count_upper;
                        if((mean_upper - median) >= curb_edge_threshold) {
                            curb_flag = 1;
                        }
                    }
                }
            }
        }
        d_curb[idx] = curb_flag;
    }
}

// =======================================================================
// Kernel to compute the sampling fraction for each unique cell.
__global__
void compute_sampling_fraction_kernel(const double* d_std, const int* d_curb, int unique_count,
                                      double min_fraction, double max_fraction, double curve_exponent,
                                      double flat_threshold, double flat_fraction,
                                      double global_min, double global_max,
                                      double* d_sampling_fraction_unique) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < unique_count) {
        double std_val = d_std[idx];
        double frac = 1.0;

        // If cell is below flat threshold, use flat_fraction.
        if(std_val < flat_threshold) {
            frac = flat_fraction;
        } else {
            // Map standard dev to fraction via exponent curve.
            double ratio = (std_val - global_min) / (global_max - global_min);
            ratio = pow(ratio, curve_exponent);
            frac = min_fraction + ratio * (max_fraction - min_fraction);
        }
        // If curb flagged, override with CURB_PERCENT.
        if(d_curb[idx]) {
            frac = CURB_PERCENT;
        }
        // Clamp fraction.
        if(frac < min_fraction) frac = min_fraction;
        if(frac > max_fraction) frac = max_fraction;

        d_sampling_fraction_unique[idx] = frac;
    }
}

// =======================================================================
// Kernel to scatter the per–unique–cell sampling fractions into a full table (size = num_cells).
__global__
void scatter_sampling_kernel(const int* d_unique_cell_ids, const double* d_sampling_fraction_unique,
                             int unique_count, int num_cells, double* d_full_sampling) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Only do the scatter. We already filled d_full_sampling with 1.0 for empty cells.
    if(i < unique_count) {
        int cid = d_unique_cell_ids[i];
        if(cid < num_cells) {
            d_full_sampling[cid] = d_sampling_fraction_unique[i];
        }
    }
}

// =======================================================================
// Main GPU decimation function.
extern "C" {
EXPORT int adaptive_decimate_points_cuda(
    const double *all_points,   // input array (n_points*3 doubles)
    int n_points,               // number of points
    int grid_cells,             // number of cells along one axis
    double min_fraction,        // minimum sampling fraction
    double max_fraction,        // maximum sampling fraction
    double curve_exponent,      // exponent for mapping variance to fraction
    double flat_threshold,      // if cell std is below this, use flat_fraction
    double flat_fraction,       // sampling fraction for flat areas
    double curb_edge_threshold, // threshold for detecting a curb (in meters)
    double **out_points,        // output: pointer to newly allocated decimated points array
    int *out_n_points           // output: number of decimated points
) {
    if(n_points <= 0)
        return -1;

    // ---------------------------------------------------------------------
    // (1) Compute bounding box on host.
    double min_x = all_points[0], max_x = all_points[0];
    double min_y = all_points[1], max_y = all_points[1];
    for (int i = 0; i < n_points; i++) {
        double x = all_points[3*i];
        double y = all_points[3*i+1];
        if(x < min_x) min_x = x;
        if(x > max_x) max_x = x;
        if(y < min_y) min_y = y;
        if(y > max_y) max_y = y;
    }
    double range_x = max_x - min_x;
    double range_y = max_y - min_y;
    double dx = (grid_cells > 0) ? range_x / grid_cells : range_x;
    double dy = (grid_cells > 0) ? range_y / grid_cells : range_y;
    if(dx <= 0 || dy <= 0) {
        // Degenerate case: return copy of all points.
        double* host_out = (double*)malloc(n_points * 3 * sizeof(double));
        if(!host_out)
            return -1;
        for (int i = 0; i < n_points * 3; i++)
            host_out[i] = all_points[i];
        *out_points = host_out;
        *out_n_points = n_points;
        return 0;
    }
    int num_cells = grid_cells * grid_cells;

    // ---------------------------------------------------------------------
    // (2) Copy input points to device.
    thrust::device_vector<double> d_points(all_points, all_points + n_points * 3);

    // ---------------------------------------------------------------------
    // (3) Compute cell IDs on GPU.
    thrust::device_vector<int> d_cell_ids(n_points);
    int blockSize = 256;
    int numBlocks = (n_points + blockSize - 1) / blockSize;
    compute_cell_ids_kernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_points.data()),
        n_points, min_x, min_y, dx, dy, grid_cells,
        thrust::raw_pointer_cast(d_cell_ids.data()));
    cudaDeviceSynchronize();

    // ---------------------------------------------------------------------
    // (4) Create an index array and sort points (and cell IDs) by cell id.
    thrust::device_vector<int> d_indices(n_points);
    thrust::sequence(d_indices.begin(), d_indices.end());
    thrust::sort_by_key(d_cell_ids.begin(), d_cell_ids.end(), d_indices.begin());

    // Create sorted points array.
    thrust::device_vector<double> d_points_sorted(n_points * 3);
    reorder_kernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_points.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        thrust::raw_pointer_cast(d_points_sorted.data()),
        n_points);
    cudaDeviceSynchronize();

    // The sorted cell IDs are now in d_cell_ids (already sorted).
    thrust::device_vector<int> d_sorted_cell_ids = d_cell_ids;

    // ---------------------------------------------------------------------
    // (5) Compute segmentation boundaries (unique cell ids and counts).
    thrust::device_vector<int> d_unique_cell_ids(n_points);
    thrust::device_vector<int> d_seg_count(n_points);
    auto new_end = thrust::reduce_by_key(
        d_sorted_cell_ids.begin(), d_sorted_cell_ids.end(),
        thrust::constant_iterator<int>(1),
        d_unique_cell_ids.begin(), d_seg_count.begin());

    int unique_count = new_end.first - d_unique_cell_ids.begin();

    // Compute exclusive scan on segment counts to get start indices.
    thrust::device_vector<int> d_seg_start(unique_count);
    thrust::exclusive_scan(d_seg_count.begin(), d_seg_count.begin() + unique_count, d_seg_start.begin());

    // ---------------------------------------------------------------------
    // (6) Launch kernel to compute per–cell statistics and curb flag.
    thrust::device_vector<double> d_std(unique_count);
    thrust::device_vector<int> d_curb(unique_count);

    numBlocks = (unique_count + blockSize - 1) / blockSize;
    per_cell_kernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_points_sorted.data()),
        thrust::raw_pointer_cast(d_seg_start.data()),
        thrust::raw_pointer_cast(d_seg_count.data()),
        thrust::raw_pointer_cast(d_unique_cell_ids.data()),
        unique_count, min_x, min_y, dx, dy, grid_cells,
        curb_edge_threshold,
        thrust::raw_pointer_cast(d_std.data()),
        thrust::raw_pointer_cast(d_curb.data()));
    cudaDeviceSynchronize();

    // ---------------------------------------------------------------------
    // (7) Compute global min and max of d_std on the host.
    thrust::host_vector<double> h_std = d_std;
    double global_min_std = 1e20, global_max_std = -1e20;
    for (int i = 0; i < unique_count; i++) {
        double s = h_std[i];
        if(s < global_min_std) global_min_std = s;
        if(s > global_max_std) global_max_std = s;
    }
    if(global_max_std == global_min_std) {
        global_max_std = global_min_std + 1e-9;
    }

    // ---------------------------------------------------------------------
    // (8) Compute sampling fraction per unique cell.
    thrust::device_vector<double> d_sampling_fraction_unique(unique_count);

    numBlocks = (unique_count + blockSize - 1) / blockSize;
    compute_sampling_fraction_kernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_std.data()),
        thrust::raw_pointer_cast(d_curb.data()),
        unique_count,
        min_fraction, max_fraction, curve_exponent,
        flat_threshold, flat_fraction,
        global_min_std, global_max_std,
        thrust::raw_pointer_cast(d_sampling_fraction_unique.data()));
    cudaDeviceSynchronize();

    // ---------------------------------------------------------------------
    // (9) Prepare a device array for full sampling fractions, default 1.0.
    thrust::device_vector<double> d_full_sampling(num_cells);
    // Use thrust::fill to set the default for all cells to 1.0.
    thrust::fill(d_full_sampling.begin(), d_full_sampling.end(), 1.0);

    // ---------------------------------------------------------------------
    // (10) Scatter the unique sampling fractions into the full table.
    numBlocks = (unique_count + blockSize - 1) / blockSize;
    scatter_sampling_kernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_unique_cell_ids.data()),
        thrust::raw_pointer_cast(d_sampling_fraction_unique.data()),
        unique_count, num_cells,
        thrust::raw_pointer_cast(d_full_sampling.data()));
    cudaDeviceSynchronize();

    // ---------------------------------------------------------------------
    // (11) For each point (in sorted order) decide whether to keep it.
    thrust::device_vector<int> d_flags(n_points);
    numBlocks = (n_points + blockSize - 1) / blockSize;
    decide_keep_kernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_sorted_cell_ids.data()),
        n_points,
        thrust::raw_pointer_cast(d_full_sampling.data()),
        thrust::raw_pointer_cast(d_flags.data()));
    cudaDeviceSynchronize();

    // ---------------------------------------------------------------------
    // (12) Exclusive scan on d_flags to compute output indices.
    thrust::device_vector<int> d_scan(n_points);
    thrust::exclusive_scan(d_flags.begin(), d_flags.end(), d_scan.begin());

    // Get final count of kept points.
    int last_flag, last_scan;
    cudaMemcpy(&last_flag, thrust::raw_pointer_cast(d_flags.data()) + (n_points - 1),
               sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_scan, thrust::raw_pointer_cast(d_scan.data()) + (n_points - 1),
               sizeof(int), cudaMemcpyDeviceToHost);

    int kept = last_flag + last_scan;

    // ---------------------------------------------------------------------
    // (13) Gather the kept points into an output array.
    thrust::device_vector<double> d_out(kept * 3);
    gather_points_kernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_points_sorted.data()),
        thrust::raw_pointer_cast(d_flags.data()),
        thrust::raw_pointer_cast(d_scan.data()),
        n_points,
        thrust::raw_pointer_cast(d_out.data()));
    cudaDeviceSynchronize();

    // ---------------------------------------------------------------------
    // (14) Copy the output points back to host.
    double* host_out = (double*)malloc(kept * 3 * sizeof(double));
    if(!host_out)
        return -1;
    cudaMemcpy(host_out, thrust::raw_pointer_cast(d_out.data()),
               kept * 3 * sizeof(double), cudaMemcpyDeviceToHost);

    *out_points = host_out;
    *out_n_points = kept;

    return 0;
}
} // extern "C"

