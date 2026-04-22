%% Diagnostic Script: Why is EM loss >> EMBR loss in EM_EMBR.m?
addpath('./functions/haversine');
addpath('./functions');
addpath('./functions/benchmarks');
addpath('./all_points');

parameters;
city = 'rome';

fprintf('=== Diagnostic: EM vs EMBR Loss Gap Analysis ===\n\n');

%% Replicate exactly the setup from EM_EMBR.m
node_file = sprintf('./datasets/%s/nodes.csv', city);
opts = detectImportOptions(node_file);
opts = setvartype(opts, 'osmid', 'int64');
df_nodes = readtable(node_file, opts);
df_edges = readtable(sprintf('./datasets/%s/edges.csv', city));

col_longitude_orig = table2array(df_nodes(:, 'x'));
col_latitude_orig  = table2array(df_nodes(:, 'y'));
NR_LOC = size(df_nodes, 1);

max_longitude = max(col_longitude_orig);
min_longitude = min(col_longitude_orig);
mid_longitude = (max_longitude+min_longitude)/2;
LONGITUDE_SIZE = max_longitude - min_longitude;

max_latitude = max(col_latitude_orig);
min_latitude = min(col_latitude_orig);
mid_latitude = (max_latitude+min_latitude)/2;
LATITUDE_SIZE = max_latitude - min_latitude;

lon_range = [mid_longitude-LONGITUDE_SIZE/SCALE, mid_longitude+LONGITUDE_SIZE/SCALE];
lat_range = [mid_latitude-LATITUDE_SIZE/SCALE, mid_latitude+LATITUDE_SIZE/SCALE];

selected_indices = filter_coords_by_range(col_longitude_orig, col_latitude_orig, lon_range, lat_range);
col_longitude = col_longitude_orig(selected_indices, :);
col_latitude  = col_latitude_orig(selected_indices, :);
[col_longitude, col_latitude] = lonlat_to_xy(col_longitude, col_latitude, mid_longitude, mid_latitude);
NR_REAL_LOC = size(col_longitude, 1);

fprintf('[City info] Filtered nodes: %d\n', NR_REAL_LOC);

%% Load path length and build per_points
path_file = sprintf('./intermediate/%s/pathLength.mat', city);
load(path_file);

perturbed_indices = 1:20;
per_points = [col_latitude(perturbed_indices', 1), col_longitude(perturbed_indices', 1)];

% Inspect per_points
fprintf('\n--- per_points (the 20 perturbed candidate locations, first 20 nodes) ---\n');
fprintf('per_points lat range: [%.4f, %.4f]\n', min(per_points(:,1)), max(per_points(:,1)));
fprintf('per_points lon range: [%.4f, %.4f]\n', min(per_points(:,2)), max(per_points(:,2)));
per_spread = max(max(pdist2(per_points, per_points)));
fprintf('Max pairwise dist within per_points: %.4f\n', per_spread);

% How do per_points relate to all nodes?
all_nodes_xy = [col_latitude, col_longitude];
dist_per_to_allnodes = pdist2(all_nodes_xy, per_points);
min_dist_each_node = min(dist_per_to_allnodes, [], 2);
fprintf('\n--- Distance from each city node to nearest per_point ---\n');
fprintf('Min: %.4f, Mean: %.4f, Median: %.4f, Max: %.4f\n', ...
    min(min_dist_each_node), mean(min_dist_each_node), median(min_dist_each_node), max(min_dist_each_node));
fprintf('Nodes farther than 2x per_points spread (%.4f): %d / %d (%.1f%%)\n', ...
    2*per_spread, sum(min_dist_each_node > 2*per_spread), NR_REAL_LOC, ...
    100*sum(min_dist_each_node > 2*per_spread)/NR_REAL_LOC);

%% Original loss matrix
loss_matrix_orig = zeros(NR_LOC, NR_PER_LOC);
for idx_real_loc = 1:NR_LOC
    for idx_per_loc = 1:NR_PER_LOC
        loss_matrix_orig(idx_real_loc, idx_per_loc) = abs(pathLength(1, idx_real_loc) - pathLength(1, perturbed_indices(idx_per_loc)));
    end
end
fprintf('\n--- loss_matrix_orig (original road-graph loss, %dx%d) ---\n', size(loss_matrix_orig,1), size(loss_matrix_orig,2));
fprintf('Min: %.2f, Mean: %.2f, Median: %.2f, Max: %.2f\n', ...
    min(loss_matrix_orig(:)), mean(loss_matrix_orig(:)), median(loss_matrix_orig(:)), max(loss_matrix_orig(:)));

%% Now inspect each layer
for layer_id = 1:3
    layers = layer_id + 1;
    fprintf('\n======= Layer %d (allpoints%d) =======\n', layer_id, layers);

    pointsname = sprintf('%s_allpoints%d_test%d.mat', city, layers, 1);
    load(pointsname);
    costname = sprintf('%s_loss_matrix_selected%d_test%d.mat', city, layers, 1);
    load(costname);

    n_all = size(allPoints, 1);
    n_per = size(per_points, 1);
    fprintf('[allPoints] Size: %dx%d\n', size(allPoints,1), size(allPoints,2));
    fprintf('[loss_matrix_selected] Size: %dx%d\n', size(loss_matrix_selected,1), size(loss_matrix_selected,2));

    % Geographic spread of allPoints vs per_points
    ap_lat = allPoints(:,1); ap_lon = allPoints(:,2);
    fprintf('[allPoints] lat: [%.4f, %.4f], lon: [%.4f, %.4f]\n', ...
        min(ap_lat), max(ap_lat), min(ap_lon), max(ap_lon));
    fprintf('[per_points] lat: [%.4f, %.4f], lon: [%.4f, %.4f]\n', ...
        min(per_points(:,1)), max(per_points(:,1)), min(per_points(:,2)), max(per_points(:,2)));

    distMat = pdist2(allPoints, per_points);
    fprintf('[distMat] Min: %.4f, Mean: %.4f, Max: %.4f\n', ...
        min(distMat(:)), mean(distMat(:)), max(distMat(:)));

    % Distance from allPoints to nearest per_point
    min_dist_ap = min(distMat, [], 2);
    fprintf('[allPoints nearest per_point] Mean: %.4f, Max: %.4f\n', mean(min_dist_ap), max(min_dist_ap));

    fprintf('[loss_matrix_selected] Min: %.2f, Mean: %.2f, Median: %.2f, Max: %.2f\n', ...
        min(loss_matrix_selected(:)), mean(loss_matrix_selected(:)), ...
        median(loss_matrix_selected(:)), max(loss_matrix_selected(:)));

    %% Compare EM vs EMBR for each epsilon
    fprintf('\n  eps     loss_EM     loss_EMBR   ratio   EM_normalized\n');
    for epsilon_idx = 1:EPSILON_MAX
        EPSILON = 0.5 * epsilon_idx * 2;

        P_matrix = zeros(n_all, n_per);
        for i = 1:n_all
            w = exp(-EPSILON * distMat(i,:) / 2.0);
            P_matrix(i,:) = w / sum(w);
        end

        cost = loss_matrix_selected / n_all;
        loss_EM = sum(sum(cost .* P_matrix));

        prior_br = ones(1, n_all) / n_all;
        BR_loss = perturbation_cal_rmp(P_matrix, prior_br, loss_matrix_selected);

        fprintf('  %.1f  %10.3f  %10.3f   %.2fx\n', EPSILON, loss_EM, BR_loss, loss_EM/max(BR_loss, 1e-9));
    end

    %% Deeper insight: WHY is EMBR so much better?
    % Check if z_star is always remapping to the SAME location (collapsing to one point)
    EPSILON = 1.0;
    P_matrix = zeros(n_all, n_per);
    for i = 1:n_all
        w = exp(-EPSILON * distMat(i,:) / 2.0);
        P_matrix(i,:) = w / sum(w);
    end
    prior_br = ones(1, n_all) / n_all;

    z_star_list = zeros(1, n_per);
    for z = 1:n_per
        post_num = prior_br .* P_matrix(:, z)';
        post = post_num / sum(post_num);
        exp_losses = post * loss_matrix_selected;
        [~, z_star_list(z)] = min(exp_losses);
    end
    fprintf('\n  [eps=1.0] z_star remap targets: ');
    fprintf('%d ', z_star_list);
    fprintf('\n');
    fprintf('  Unique z_star values: %d (out of %d possible per_points)\n', numel(unique(z_star_list)), n_per);

    % What does EM report on average vs what's the best choice?
    % Compute the "oracle" loss: for each true loc i, what is the minimum possible loss?
    min_loss_per_true = min(loss_matrix_selected, [], 2);
    oracle_loss = sum(min_loss_per_true) / n_all;
    fprintf('  Oracle loss (best fixed report per true loc): %.3f\n', oracle_loss);

    % EM loss at eps=1
    EPSILON = 1.0;
    P_matrix = zeros(n_all, n_per);
    for i = 1:n_all
        w = exp(-EPSILON * distMat(i,:) / 2.0);
        P_matrix(i,:) = w / sum(w);
    end
    cost = loss_matrix_selected / n_all;
    loss_EM_1 = sum(sum(cost .* P_matrix));

    % What is the "collapse" loss: always report z_star=1 for all z?
    % This tests whether EMBR just collapses everything to one location
    if numel(unique(z_star_list)) == 1
        z_single = z_star_list(1);
        collapse_loss = sum(loss_matrix_selected(:, z_single)) / n_all;
        fprintf('  All z_star == %d! Collapse loss: %.3f (EM loss: %.3f)\n', z_single, collapse_loss, loss_EM_1);
        fprintf('  => EMBR trivially minimizes by always remapping to per_point %d\n', z_single);
    end
end

fprintf('\n=== Done ===\n');
