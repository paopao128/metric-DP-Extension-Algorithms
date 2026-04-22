%% Diagnostic: Why does loss_aipo_instance_approx differ so much from loss_aipo_instance?
% This script traces through the computation and isolates the sources of discrepancy.

addpath('./functions/haversine');
addpath('./functions');
addpath('./functions/benchmarks');

parameters;
city_list = {'rome', 'nyc', 'london'};
l_p = 2;

city_idx = 1;
city = city_list{city_idx};
fprintf('City: %s\n', city);

%% Setup (same as main)
node_file = sprintf('./datasets/%s/nodes.csv', city);
edge_file = sprintf('./datasets/%s/edges.csv', city);
opts = detectImportOptions(node_file);
opts = setvartype(opts, 'osmid', 'int64');
df_nodes = readtable(node_file, opts);
df_edges = readtable(edge_file);

col_longitude_orig = table2array(df_nodes(:, 'x'));
col_latitude_orig  = table2array(df_nodes(:, 'y'));
col_osmid = table2array(df_nodes(:, 'osmid'));

max_longitude = max(col_longitude_orig); min_longitude = min(col_longitude_orig);
mid_longitude = (max_longitude+min_longitude)/2;
LONGITUDE_SIZE = max_longitude - min_longitude;

max_latitude = max(col_latitude_orig); min_latitude = min(col_latitude_orig);
mid_latitude = (max_latitude+min_latitude)/2;
LATITUDE_SIZE = max_latitude - min_latitude;

lon_range = [mid_longitude-LONGITUDE_SIZE/SCALE, mid_longitude+LONGITUDE_SIZE/SCALE];
lat_range = [mid_latitude-LATITUDE_SIZE/SCALE,   mid_latitude+LATITUDE_SIZE/SCALE];

selected_indices = filter_coords_by_range(col_longitude_orig, col_latitude_orig, lon_range, lat_range);
col_longitude = col_longitude_orig(selected_indices, :);
col_latitude  = col_latitude_orig(selected_indices, :);
[col_longitude, col_latitude] = lonlat_to_xy(col_longitude, col_latitude, mid_longitude, mid_latitude);
NR_REAL_LOC = size(col_longitude, 1);

prior = ones(1, NR_REAL_LOC)/NR_REAL_LOC;

rng(42);  % Fix random seed for reproducibility
samples_viocheck = randperm(NR_REAL_LOC, SAMPLE_SIZE_PPR);
perturbed_indices = randi([1, NR_REAL_LOC], 1, NR_PER_LOC);
perturbed_xy = [col_latitude(perturbed_indices', 1), col_longitude(perturbed_indices', 1)];
loss_matrix = pdist2([col_latitude, col_longitude], perturbed_xy);
loss_matrix_max = min(loss_matrix, [], 2);

[adjMatrix, distanceMatrix, neighborMatrix, cornerPoints_ori, squares, lambda_x, lambda_y, corner_weights] = ...
    uniform_anchor(col_latitude, col_longitude, loss_matrix_max, cell_size(1, city_idx));
close all;

selected_idx_inner = 1:NR_REAL_LOC;
loss_matrix_selected = loss_matrix(selected_idx_inner, :);
corner_weights_selected_ori = corner_weights(selected_idx_inner, :);

epsilon_idx = 1;
EPSILON = 0.5 * epsilon_idx;
epsilon_idx_1 = 2;  % Fix one budget split to examine
epsilon_1 = EPSILON * epsilon_idx_1 / NR_EPSILON_INTERVAL;
epsilon_2 = EPSILON * sqrt(1 - (epsilon_idx_1/NR_EPSILON_INTERVAL)^2);

corner_weights_selected = corner_weights_selected_ori;
cornerPoints = cornerPoints_ori;

%% ===== Compute c_approx and LP solution (same as lines 97, 107-108) =====
c_approx = corner_weights_selected' * loss_matrix_selected;

fprintf('\n============ SIZES ============\n');
fprintf('NR_REAL_LOC          = %d  (real locations)\n', NR_REAL_LOC);
fprintf('NR_PER_LOC           = %d  (perturbed candidates)\n', NR_PER_LOC);
fprintf('NR_ANCHOR (corners)  = %d  (anchor grid points)\n', size(cornerPoints_ori, 1));
fprintf('c_approx size        = %d x %d\n', size(c_approx));
fprintf('corner_weights size  = %d x %d\n', size(corner_weights_selected));

[z_anchor_instance, fval_raw] = perturbation_cal_apo(c_approx, corner_weights_selected, distanceMatrix, neighborMatrix, epsilon_1/2, epsilon_2/2);
loss_approx_raw    = fval_raw;
loss_approx_norm   = fval_raw / NR_REAL_LOC;  % line 108 in main

fprintf('\n============ loss_aipo_instance_approx BREAKDOWN ============\n');
fprintf('LP objective (fval_raw)         = %.6f\n', fval_raw);
fprintf('After /NR_REAL_LOC (line 108)   = %.6f  [this is loss_aipo_instance_approx]\n', loss_approx_norm);

%% --- Verify LP objective manually ---
% fval should equal sum_{i,k} c_approx(i,k)*z_anchor_instance(i,k)
fval_manual = sum(sum(c_approx .* z_anchor_instance));
fprintf('Manual recompute of LP obj      = %.6f  (should match fval_raw)\n', fval_manual);

% Expand to real locations via bilinear interpolation
x_bilinear = corner_weights_selected * z_anchor_instance;  % NR_REAL_LOC x NR_PER_LOC
loss_bilinear_real = sum(sum(x_bilinear .* loss_matrix_selected)) / NR_REAL_LOC;
fprintf('Bilinear-interp loss at REAL locs = %.6f\n', loss_bilinear_real);

%% ===== Run the interpolation loop (same as lines 112-126) =====
interp_times = 3;
allPoints = cornerPoints_ori;
loss_matrix_selected_cur = loss_matrix_selected;
corner_weights_selected_cur = corner_weights_selected_ori;
z_cur = z_anchor_instance;

fprintf('\n============ loss_aipo_instance BREAKDOWN (interp loop) ============\n');
for interp_id = 1:interp_times
    cornerPoints = allPoints;
    interpolation_extension_k;  % produces W, allPoints
    corner_weights_selected = W;
    loss_matrix_selected = pdist2(allPoints, perturbed_xy);
    if interp_id > 1
        z_cur = z_extension;
    end
    [z_extension, loss_interp] = logconv_interp(z_cur, corner_weights_selected, loss_matrix_selected);

    fprintf('  Iter %d: allPoints=%d, loss_matrix_selected=%dx%d, loss=%.6f\n', ...
        interp_id, size(allPoints,1), size(loss_matrix_selected,1), size(loss_matrix_selected,2), loss_interp);
end
loss_final = loss_interp;

fprintf('\n============ COMPARISON SUMMARY ============\n');
fprintf('loss_aipo_instance_approx  = %.6f  (LP obj / NR_REAL_LOC=%d)\n', loss_approx_norm, NR_REAL_LOC);
fprintf('loss_aipo_instance (final) = %.6f  (logconv_interp / nAllPoints=%d)\n', loss_final, size(allPoints,1));
fprintf('Ratio (approx/final)       = %.4f\n', loss_approx_norm / loss_final);

%% ===== Isolate hypotheses =====
fprintf('\n============ HYPOTHESIS TESTING ============\n');

% H1: Different denominators?
fprintf('H1 - Denominator difference:\n');
fprintf('  approx denominator = NR_REAL_LOC = %d\n', NR_REAL_LOC);
fprintf('  final  denominator = size(allPoints,1) = %d\n', size(allPoints,1));
fprintf('  If approx used same denominator: %.6f\n', fval_raw / size(allPoints,1));

% H2: If logconv_interp applied to ORIGINAL anchors (no refinement) vs. after refinement
[~, loss_logconv_at_orig] = logconv_interp(z_anchor_instance, corner_weights_selected_ori, loss_matrix(selected_idx_inner,:));
fprintf('\nH2 - logconv at ORIGINAL real locs (no refine): %.6f\n', loss_logconv_at_orig);

% H3: c_approx quality - compare LP objective with actual log-conv loss at anchors
% The LP uses c_approx to approximate cost. Let's see what the true cost is
% at anchor points under the LP solution
loss_at_anchors_direct = sum(sum(z_anchor_instance .* c_approx)) / size(cornerPoints_ori,1);
fprintf('\nH3 - LP cost normalized by NR_ANCHOR: %.6f\n', loss_at_anchors_direct);

% H4: Log-convex interpolation at ANCHOR GRID (not fine grid), same denominator
corner_pts_loss = pdist2(cornerPoints_ori, perturbed_xy);
[~, loss_logconv_anchors] = logconv_interp(z_anchor_instance, corner_weights_selected_ori, corner_pts_loss);
fprintf('H4 - logconv at ANCHOR GRID pts (denom=%d): %.6f\n', size(cornerPoints_ori,1), loss_logconv_anchors);

% H5: Compare z_anchor LP solution vs uniform distribution
z_uniform = ones(size(z_anchor_instance)) / size(z_anchor_instance, 2);
loss_uniform_approx = sum(sum(c_approx .* z_uniform)) / NR_REAL_LOC;
fprintf('\nH5 - Uniform distribution loss (approx): %.6f\n', loss_uniform_approx);

fprintf('\nDone.\n');
