%% Diagnose spatial mismatch between fine grid points and real locations
addpath('./functions/haversine'); addpath('./functions'); addpath('./functions/benchmarks');
parameters;
city_list = {'rome'}; l_p = 2; city_idx = 1; city = city_list{city_idx};
node_file = sprintf('./datasets/%s/nodes.csv', city);
opts = detectImportOptions(node_file); opts = setvartype(opts, 'osmid', 'int64');
df_nodes = readtable(node_file, opts);
col_longitude_orig = table2array(df_nodes(:, 'x')); col_latitude_orig = table2array(df_nodes(:, 'y'));
max_longitude = max(col_longitude_orig); min_longitude = min(col_longitude_orig); mid_longitude = (max_longitude+min_longitude)/2; LONGITUDE_SIZE = max_longitude - min_longitude;
max_latitude = max(col_latitude_orig); min_latitude = min(col_latitude_orig); mid_latitude = (max_latitude+min_latitude)/2; LATITUDE_SIZE = max_latitude - min_latitude;
lon_range = [mid_longitude-LONGITUDE_SIZE/SCALE, mid_longitude+LONGITUDE_SIZE/SCALE];
lat_range = [mid_latitude-LATITUDE_SIZE/SCALE,   mid_latitude+LATITUDE_SIZE/SCALE];
selected_indices = filter_coords_by_range(col_longitude_orig, col_latitude_orig, lon_range, lat_range);
col_longitude = col_longitude_orig(selected_indices,:); col_latitude = col_latitude_orig(selected_indices,:);
[col_longitude, col_latitude] = lonlat_to_xy(col_longitude, col_latitude, mid_longitude, mid_latitude);
NR_REAL_LOC = size(col_longitude,1);
rng(42);
perturbed_indices = randi([1, NR_REAL_LOC], 1, NR_PER_LOC);
perturbed_xy = [col_latitude(perturbed_indices',1), col_longitude(perturbed_indices',1)];
loss_matrix = pdist2([col_latitude, col_longitude], perturbed_xy);
loss_matrix_max = min(loss_matrix,[],2);
[adjMatrix, distanceMatrix, neighborMatrix, cornerPoints_ori, squares, ~, ~, corner_weights] = ...
    uniform_anchor(col_latitude, col_longitude, loss_matrix_max, cell_size(1, city_idx));
close all;
NR_ANCHOR = size(cornerPoints_ori,1);

% Run 3 rounds of k=2 interpolation to get fine grid
cornerPoints = cornerPoints_ori; allPoints = cornerPoints_ori;
for it = 1:3
    cornerPoints = allPoints;
    interpolation_extension_k;  % updates allPoints, W
end
nFineGrid = size(allPoints, 1);

% For each fine grid point, find distance to nearest real location
real_locs = [col_latitude, col_longitude];
dist_to_nearest_real = min(pdist2(allPoints, real_locs), [], 2);

% Threshold = half the original cell size (points far from any real loc are "empty")
thresh = cell_size(1, city_idx) / 2;
n_far = sum(dist_to_nearest_real > thresh);
n_near = sum(dist_to_nearest_real <= thresh);

fprintf('\n===== SPATIAL MISMATCH ANALYSIS =====\n');
fprintf('Original anchor grid points:       %d\n', NR_ANCHOR);
fprintf('Fine grid points (3 interp rounds): %d\n', nFineGrid);
fprintf('Real locations (road nodes):        %d\n', NR_REAL_LOC);
fprintf('\nThreshold for "near a real loc":    %.2f km\n', thresh);
fprintf('Fine grid pts near a real loc:      %d (%.1f%%)\n', n_near, 100*n_near/nFineGrid);
fprintf('Fine grid pts FAR from real locs:   %d (%.1f%%)\n', n_far, 100*n_far/nFineGrid);

% Expected loss under uniform perturbation (shows inherent difficulty by location)
avg_loss_per_pt = mean(pdist2(allPoints, perturbed_xy), 2);
fprintf('\nUniform-perturbation avg expected loss:\n');
fprintf('  At NEAR fine grid pts:  %.4f km\n', mean(avg_loss_per_pt(dist_to_nearest_real <= thresh)));
fprintf('  At FAR  fine grid pts:  %.4f km\n', mean(avg_loss_per_pt(dist_to_nearest_real > thresh)));
fprintf('  Overall fine grid avg:  %.4f km\n', mean(avg_loss_per_pt));
avg_loss_real = mean(mean(pdist2(real_locs, perturbed_xy), 2));
fprintf('  At REAL locations:      %.4f km\n', avg_loss_real);

% Compute what loss_aipo_instance_approx vs loss_aipo_instance look like
% under the same (uniform) perturbation distribution for fair comparison
c_approx = corner_weights(1:NR_REAL_LOC,:)' * loss_matrix;
z_uniform_anchor = ones(NR_ANCHOR, NR_PER_LOC) / NR_PER_LOC;
loss_approx_uniform = sum(sum(c_approx .* z_uniform_anchor)) / NR_REAL_LOC;
fprintf('\nUnder UNIFORM perturbation distribution:\n');
fprintf('  loss_approx formula (real locs):  %.4f km\n', loss_approx_uniform);
fprintf('  Direct avg over real locs:        %.4f km\n', avg_loss_real);

% logconv_interp with uniform anchor = uniform fine grid
[~, loss_logconv_uniform] = logconv_interp(z_uniform_anchor, W, pdist2(allPoints, perturbed_xy));
fprintf('  loss_logconv formula (fine grid): %.4f km\n', loss_logconv_uniform);
fprintf('  Direct avg over fine grid:        %.4f km\n', mean(avg_loss_per_pt));

fprintf('\n=> Summary: The ~3x gap is because loss_aipo_instance_approx averages\n');
fprintf('   over %d real road nodes (clustered near streets), while\n', NR_REAL_LOC);
fprintf('   loss_aipo_instance averages over %d uniform grid points\n', nFineGrid);
fprintf('   (including ~%.0f%% of points far from any road).\n', 100*n_far/nFineGrid);
fprintf('   Empty-area grid points have higher expected loss, inflating the average.\n');
