// decimation.c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

// Parameters for curb detection
#define MIN_POINTS_FOR_CURB 12      // Minimum points in a (sub)cell to perform curb detection
#define MIN_UPPER_POINTS 8          // Minimum points in the upper group to flag a curb
#define SUBDIV_FACTOR 4             // Subdivide each main cell into SUBDIV_FACTOR x SUBDIV_FACTOR sub-cells
#define CURB_PERCENT 0.2	        // The percent of points that will remain in a cell that was considered a curb

// qsort comparison function for doubles.
static int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    else if (da > db) return 1;
    else return 0;
}

// Helper: Solve a 3x3 linear system M*x = b using Gaussian elimination.
// Returns 0 on success and -1 if the system is singular.
static int solve_3x3(double M[3][3], double b[3], double x[3]) {
    int i, j, k;
    for (i = 0; i < 3; i++) {
        // Find pivot
        double max_val = fabs(M[i][i]);
        int pivot = i;
        for (j = i+1; j < 3; j++) {
            double val = fabs(M[j][i]);
            if (val > max_val) {
                max_val = val;
                pivot = j;
            }
        }
        if (max_val < 1e-12) return -1; // singular
        // Swap rows if necessary
        if (pivot != i) {
            for (k = 0; k < 3; k++) {
                double temp = M[i][k];
                M[i][k] = M[pivot][k];
                M[pivot][k] = temp;
            }
            double temp = b[i];
            b[i] = b[pivot];
            b[pivot] = temp;
        }
        // Elimination
        for (j = i+1; j < 3; j++) {
            double factor = M[j][i] / M[i][i];
            for (k = i; k < 3; k++) {
                M[j][k] -= factor * M[i][k];
            }
            b[j] -= factor * b[i];
        }
    }
    // Back substitution
    for (i = 2; i >= 0; i--) {
        if (fabs(M[i][i]) < 1e-12) return -1;
        x[i] = b[i];
        for (j = i+1; j < 3; j++) {
            x[i] -= M[i][j] * x[j];
        }
        x[i] /= M[i][i];
    }
    return 0;
}

///
/// adaptive_decimate_points
///
/// Parameters:
///   - all_points: pointer to an array of n_points*3 doubles (each point: x, y, z)
///   - n_points: number of points in the array
///   - grid_cells: number of cells along one axis (total cells = grid_cells*grid_cells)
///   - min_fraction, max_fraction, curve_exponent, flat_threshold, flat_fraction: parameters controlling decimation
///   - curb_edge_threshold: if (in any sub-cell) the difference between the median z and the average of points above the median
///                          is >= this threshold (in meters) and the upper group contains at least MIN_UPPER_POINTS,
///                          then that sub-cell is flagged and the main cell is considered to contain a curb.
///   - out_points: pointer to a pointer which will be set to a newly allocated array of decimated points (each point has 3 doubles)
///   - out_n_points: pointer to an integer which will receive the number of decimated points
///
/// Returns 0 on success or -1 on error.
EXPORT int adaptive_decimate_points(
    const double* all_points, // input array (n_points*3)
    int n_points,
    int grid_cells,
    double min_fraction,
    double max_fraction,
    double curve_exponent,
    double flat_threshold,
    double flat_fraction,
    double curb_edge_threshold,  // threshold (in meters) for detecting a curb in a sub-cell
    double** out_points,      // output: newly allocated array of decimated points (size out_n_points*3)
    int* out_n_points         // output: number of decimated points
) {
    if(n_points <= 0) return -1;
    
    // Determine x,y bounds.
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
    double dx = (grid_cells > 0) ? (max_x - min_x) / grid_cells : (max_x - min_x);
    double dy = (grid_cells > 0) ? (max_y - min_y) / grid_cells : (max_y - min_y);
    if(dx <= 0 || dy <= 0) {
        // If invalid dimensions, return a copy of all points.
        *out_points = (double*)malloc(n_points * 3 * sizeof(double));
        if(!*out_points) return -1;
        for (int i = 0; i < n_points * 3; i++) {
            (*out_points)[i] = all_points[i];
        }
        *out_n_points = n_points;
        return 0;
    }
    
    // Compute cell indices for each point.
    int* cell_ids = (int*)malloc(n_points * sizeof(int));
    if(!cell_ids) return -1;
    for (int i = 0; i < n_points; i++) {
        double x = all_points[3*i];
        double y = all_points[3*i+1];
        int ix = (int)((x - min_x) / dx);
        int iy = (int)((y - min_y) / dy);
        if(ix < 0) ix = 0; if(ix >= grid_cells) ix = grid_cells - 1;
        if(iy < 0) iy = 0; if(iy >= grid_cells) iy = grid_cells - 1;
        cell_ids[i] = iy * grid_cells + ix;
    }
    
    int num_cells = grid_cells * grid_cells;
    // Count points per cell.
    int* cell_counts = (int*)calloc(num_cells, sizeof(int));
    if(!cell_counts) { free(cell_ids); return -1; }
    for (int i = 0; i < n_points; i++) {
        cell_counts[cell_ids[i]]++;
    }
    
    // Allocate arrays to hold point indices for each cell.
    int** cell_indices = (int**)malloc(num_cells * sizeof(int*));
    if(!cell_indices) { free(cell_ids); free(cell_counts); return -1; }
    for (int c = 0; c < num_cells; c++) {
        if(cell_counts[c] > 0) {
            cell_indices[c] = (int*)malloc(cell_counts[c] * sizeof(int));
            if(!cell_indices[c]) {
                for (int k = 0; k < c; k++) {
                    free(cell_indices[k]);
                }
                free(cell_indices); free(cell_ids); free(cell_counts);
                return -1;
            }
        } else {
            cell_indices[c] = NULL;
        }
    }
    
    // Populate each cell's indices.
    int* current_index = (int*)calloc(num_cells, sizeof(int));
    if(!current_index) {
        for (int c = 0; c < num_cells; c++) {
            if(cell_indices[c]) free(cell_indices[c]);
        }
        free(cell_indices); free(cell_ids); free(cell_counts);
        return -1;
    }
    for (int i = 0; i < n_points; i++) {
        int c = cell_ids[i];
        cell_indices[c][ current_index[c]++ ] = i;
    }
    
    // Allocate array for computed standard deviations for each cell.
    double* cell_std = (double*)malloc(num_cells * sizeof(double));
    if(!cell_std) {
        free(current_index);
        for (int c = 0; c < num_cells; c++) { if(cell_indices[c]) free(cell_indices[c]); }
        free(cell_indices); free(cell_ids); free(cell_counts);
        return -1;
    }
    
    // For each cell, compute the best-fit plane (if possible) and standard deviation.
    for (int c = 0; c < num_cells; c++) {
        int count = cell_counts[c];
        if(count < 3) {
            cell_std[c] = 0.0;
        } else {
            double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
            double sum_xx = 0.0, sum_xy = 0.0, sum_yy = 0.0;
            double sum_xz = 0.0, sum_yz = 0.0;
            for (int j = 0; j < count; j++) {
                int i = cell_indices[c][j];
                double x = all_points[3*i];
                double y = all_points[3*i+1];
                double z = all_points[3*i+2];
                sum_x += x;  sum_y += y;  sum_z += z;
                sum_xx += x*x;  sum_xy += x*y;  sum_yy += y*y;
                sum_xz += x*z;  sum_yz += y*z;
            }
            double M[3][3] = {
                { sum_xx, sum_xy, sum_x },
                { sum_xy, sum_yy, sum_y },
                { sum_x,  sum_y,  (double)count }
            };
            double B[3] = { sum_xz, sum_yz, sum_z };
            double coeff[3] = {0.0, 0.0, 0.0};
            if(solve_3x3(M, B, coeff) != 0) {
                cell_std[c] = 0.0;
            } else {
                double a = coeff[0], b = coeff[1], c_term = coeff[2];
                double sum_res2 = 0.0;
                for (int j = 0; j < count; j++) {
                    int i = cell_indices[c][j];
                    double x = all_points[3*i];
                    double y = all_points[3*i+1];
                    double z = all_points[3*i+2];
                    double z_pred = a * x + b * y + c_term;
                    double r = z - z_pred;
                    sum_res2 += r * r;
                }
                cell_std[c] = sqrt(sum_res2 / count);
            }
        }
    }
    
    // For each cell, perform curb detection using sub-grid (sub-cell) analysis.
    // We subdivide the main cell into SUBDIV_FACTOR x SUBDIV_FACTOR sub-cells and perform a median-based detection in each.
    int* is_curb = (int*)calloc(num_cells, sizeof(int));
    if(!is_curb) {
        free(cell_std); free(current_index);
        for (int c = 0; c < num_cells; c++) { if(cell_indices[c]) free(cell_indices[c]); }
        free(cell_indices); free(cell_ids); free(cell_counts);
        return -1;
    }
    for (int c = 0; c < num_cells; c++) {
        int count = cell_counts[c];
        if(count < MIN_POINTS_FOR_CURB) {
            is_curb[c] = 0;
            continue;
        }
        // Compute the boundaries of the main cell.
        int row = c / grid_cells;
        int col = c % grid_cells;
        double cell_x_min = min_x + col * dx;
        double cell_y_min = min_y + row * dy;
        double cell_x_max = cell_x_min + dx;
        double cell_y_max = cell_y_min + dy;
        int curb_flag = 0;
        // Subdivide the cell.
        double sub_dx = dx / SUBDIV_FACTOR;
        double sub_dy = dy / SUBDIV_FACTOR;
        for (int sr = 0; sr < SUBDIV_FACTOR; sr++) {
            for (int sc = 0; sc < SUBDIV_FACTOR; sc++) {
                double sub_x_min = cell_x_min + sc * sub_dx;
                double sub_y_min = cell_y_min + sr * sub_dy;
                double sub_x_max = sub_x_min + sub_dx;
                double sub_y_max = sub_y_min + sub_dy;
                // Collect z-values for points in this sub-cell.
                int sub_count = 0;
                double* z_vals = (double*)malloc(count * sizeof(double));
                if(!z_vals) continue; // if allocation fails, skip this sub-cell
                for (int j = 0; j < count; j++) {
                    int i = cell_indices[c][j];
                    double x = all_points[3*i];
                    double y = all_points[3*i+1];
                    if(x >= sub_x_min && x < sub_x_max && y >= sub_y_min && y < sub_y_max) {
                        z_vals[sub_count++] = all_points[3*i+2];
                    }
                }
                if(sub_count >= MIN_POINTS_FOR_CURB) {
                    // Sort the sub-cell z-values.
                    qsort(z_vals, sub_count, sizeof(double), compare_doubles);
                    double median = z_vals[sub_count/2];
                    // Compute the average of points above the median.
                    double sum_upper = 0.0;
                    int count_upper = 0;
                    for (int j = 0; j < sub_count; j++) {
                        if(z_vals[j] > median) {
                            sum_upper += z_vals[j];
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
                free(z_vals);
                if(curb_flag) break;
            }
            if(curb_flag) break;
        }
        is_curb[c] = curb_flag;
    }
    
    // Compute a sampling fraction for each cell.
    double* cell_fraction = (double*)malloc(num_cells * sizeof(double));
    if(!cell_fraction) {
        free(is_curb); free(cell_std); free(current_index);
        for (int c = 0; c < num_cells; c++) { if(cell_indices[c]) free(cell_indices[c]); }
        free(cell_indices); free(cell_ids); free(cell_counts);
        return -1;
    }
    for (int c = 0; c < num_cells; c++) {
        double frac;
        if(is_curb[c]) {
            // If flagged as curb, keep 100% of points.
            frac = CURB_PERCENT;
        } else if(cell_std[c] < flat_threshold) {
            frac = flat_fraction;
        } else {
            // Otherwise, map the standard deviation (normalized over all cells) to a fraction.
            double global_min_std = cell_std[0], global_max_std = cell_std[0];
            for (int cc = 0; cc < num_cells; cc++) {
                if(cell_std[cc] < global_min_std) global_min_std = cell_std[cc];
                if(cell_std[cc] > global_max_std) global_max_std = cell_std[cc];
            }
            if(global_min_std == global_max_std) {
                global_max_std = global_min_std + 1e-9;
            }
            double ratio = (cell_std[c] - global_min_std) / (global_max_std - global_min_std);
            ratio = pow(ratio, curve_exponent);
            frac = min_fraction + ratio * (max_fraction - min_fraction);
            if(frac < min_fraction) frac = min_fraction;
            if(frac > max_fraction) frac = max_fraction;
        }
        cell_fraction[c] = frac;
    }
    
    // For each point, decide whether to keep it based on its cell's sampling fraction.
    int kept_count = 0;
    int* keep_mask = (int*)malloc(n_points * sizeof(int));
    if(!keep_mask) {
        free(cell_fraction); free(is_curb); free(cell_std); free(current_index);
        for (int c = 0; c < num_cells; c++) { if(cell_indices[c]) free(cell_indices[c]); }
        free(cell_indices); free(cell_ids); free(cell_counts);
        return -1;
    }
    for (int i = 0; i < n_points; i++) {
        int c = cell_ids[i];
        double frac = cell_fraction[c];
        double r = ((double)rand()) / ((double)RAND_MAX);
        if(r < frac) {
            keep_mask[i] = 1;
            kept_count++;
        } else {
            keep_mask[i] = 0;
        }
    }
    
    // Allocate output array for the decimated points.
    double* decimated = (double*)malloc(kept_count * 3 * sizeof(double));
    if(!decimated) {
        free(keep_mask); free(cell_fraction); free(is_curb); free(cell_std); free(current_index);
        for (int c = 0; c < num_cells; c++) { if(cell_indices[c]) free(cell_indices[c]); }
        free(cell_indices); free(cell_ids); free(cell_counts);
        return -1;
    }
    int pos = 0;
    for (int i = 0; i < n_points; i++) {
        if(keep_mask[i]) {
            decimated[3*pos]     = all_points[3*i];
            decimated[3*pos + 1] = all_points[3*i+1];
            decimated[3*pos + 2] = all_points[3*i+2];
            pos++;
        }
    }
    
    // Set outputs.
    *out_points = decimated;
    *out_n_points = kept_count;
    
    // Free temporary allocations.
    free(keep_mask);
    free(cell_fraction);
    free(is_curb);
    free(cell_std);
    free(current_index);
    for (int c = 0; c < num_cells; c++) {
        if(cell_indices[c]) free(cell_indices[c]);
    }
    free(cell_indices);
    free(cell_ids);
    free(cell_counts);
    
    return 0;
}
