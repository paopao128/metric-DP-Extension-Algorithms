%% Diagnostic 2: Compare per_points=first20 vs random selection, and check path-length coverage
addpath('./functions/haversine');
addpath('./functions');
addpath('./functions/benchmarks');
addpath('./all_points');

parameters;
city = 'rome';

fprintf('=== Diagnostic 2: per_points selection impact ===\n\n');

%% Load city data (same as EM_EMBR.m)
node_file = sprintf('./datasets/%s/nodes.csv', city);
opts = detectImportOptions(node_file);
opts = setvartype(opts, 'osmid', 'int64');
df_nodes = readtable(node_file, opts);

col_longitude_orig = table2array(df_nodes(:, 'x'));
col_latitude_orig  = table2array(df_nodes(:, 'y'));
NR_LOC = size(df_nodes, 1);

max_longitude = max(col_longitude_orig); min_longitude = min(col_longitude_orig);
mid_longitude = (max_longitude+min_longitude)/2; LONGITUDE_SIZE = max_longitude - min_longitude;
max_latitude  = max(col_latitude_orig);  min_latitude  = min(col_latitude_orig);
mid_latitude  = (max_latitude+min_latitude)/2;  LATITUDE_SIZE  = max_latitude  - min_latitude;

lon_range = [mid_longitude-LONGITUDE_SIZE/SCALE, mid_longitude+LONGITUDE_SIZE/SCALE];
lat_range = [mid_latitude-LATITUDE_SIZE/SCALE,   mid_latitude+LATITUDE_SIZE/SCALE];
selected_indices = filter_coords_by_range(col_longitude_orig, col_latitude_orig, lon_range, lat_range);
col_longitude = col_longitude_orig(selected_indices, :);
col_latitude  = col_latitude_orig(selected_indices, :);
[col_longitude, col_latitude] = lonlat_to_xy(col_longitude, col_latitude, mid_longitude, mid_latitude);
NR_REAL_LOC = size(col_longitude, 1);

path_file = sprintf('./intermediate/%s/pathLength.mat', city);
load(path_file);

%% Load layer 1 allPoints for the test
layer_id = 1; layers = layer_id + 1;
pointsname = sprintf('%s_allpoints%d_test%d.mat', city, layers, 1);
load(pointsname);  % loads allPoints
n_all = size(allPoints, 1);

fprintf('[Setup] NR_REAL_LOC=%d, allPoints size=%d\n', NR_REAL_LOC, n_all);

%% === Scenario 1: per_points = first 20 nodes (as in EM_EMBR.m) ===
fprintf('\n--- Scenario 1: per_points = first 20 nodes (current EM_EMBR.m) ---\n');
perturbed_indices_1 = 1:20;
per_points_1 = [col_latitude(perturbed_indices_1', 1), col_longitude(perturbed_indices_1', 1)];

% Build loss matrix for scenario 1
loss_mat_1 = zeros(NR_LOC, 20);
for ii = 1:NR_LOC
    for jj = 1:20
        loss_mat_1(ii, jj) = abs(pathLength(1, ii) - pathLength(1, perturbed_indices_1(jj)));
    end
end
% For allPoints, use the loss_matrix_selected from file
costname = sprintf('%s_loss_matrix_selected%d_test%d.mat', city, layers, 1);
load(costname);  % loads loss_matrix_selected (n_all x 20)
loss_mat_1_all = loss_matrix_selected;

fprintf('per_points path lengths: '); fprintf('%.0f ', pathLength(1, perturbed_indices_1)); fprintf('\n');
fprintf('All path lengths: min=%.0f, max=%.0f, mean=%.0f\n', ...
    min(pathLength), max(pathLength), mean(pathLength));
fprintf('per_points path length coverage: [%.0f, %.0f]\n', ...
    min(pathLength(1, perturbed_indices_1)), max(pathLength(1, perturbed_indices_1)));

% Geographic spread
fprintf('per_points geo spread (max pairwise dist): %.4f\n', max(max(pdist2(per_points_1, per_points_1))));
fprintf('allPoints geo spread (max pairwise dist): %.4f\n', max(max(pdist2(allPoints(1:min(500,end),:), allPoints(1:min(500,end),:)))));

%% === Scenario 2: per_points = random 20 nodes (like main_2norm.m) ===
fprintf('\n--- Scenario 2: per_points = random 20 nodes (like main_2norm.m) ---\n');
rng(42);  % reproducible
perturbed_indices_2 = randi([1, NR_REAL_LOC], 1, 20);
per_points_2 = [col_latitude(perturbed_indices_2', 1), col_longitude(perturbed_indices_2', 1)];

% Build loss matrix for scenario 2 (allPoints x 20)
loss_mat_2_all = zeros(n_all, 20);
% Note: allPoints doesn't have direct pathLength, so we approximate using nearest node
% For simplicity, map each allPoint to nearest road node
all_nodes_xy = [col_latitude, col_longitude];
dist_ap_to_nodes = pdist2(allPoints, all_nodes_xy);
[~, nearest_node_idx] = min(dist_ap_to_nodes, [], 2);  % for each allPoint, nearest node index

for ii = 1:n_all
    node_ii = nearest_node_idx(ii);
    for jj = 1:20
        loss_mat_2_all(ii, jj) = abs(pathLength(1, node_ii) - pathLength(1, perturbed_indices_2(jj)));
    end
end

fprintf('per_points path lengths: '); fprintf('%.0f ', pathLength(1, perturbed_indices_2)); fprintf('\n');
fprintf('per_points path length coverage: [%.0f, %.0f]\n', ...
    min(pathLength(1, perturbed_indices_2)), max(pathLength(1, perturbed_indices_2)));
fprintf('per_points geo spread (max pairwise dist): %.4f\n', max(max(pdist2(per_points_2, per_points_2))));

%% === Compare EM vs EMBR for both scenarios ===
EPSILON = 1.0;

% Scenario 1
distMat_1 = pdist2(allPoints, per_points_1);
P_mat_1 = zeros(n_all, 20);
for i = 1:n_all
    w = exp(-EPSILON * distMat_1(i,:) / 2.0);
    P_mat_1(i,:) = w / sum(w);
end
cost_1 = loss_mat_1_all / n_all;
loss_EM_1 = sum(sum(cost_1 .* P_mat_1));
prior_1 = ones(1, n_all) / n_all;
BR_1 = perturbation_cal_rmp(P_mat_1, prior_1, loss_mat_1_all);
fprintf('\n[eps=%.1f] Scenario 1 (first 20 nodes):\n', EPSILON);
fprintf('  loss_EM=%.1f, loss_EMBR=%.1f, ratio=%.2fx\n', loss_EM_1, BR_1, loss_EM_1/BR_1);

% Scenario 2
distMat_2 = pdist2(allPoints, per_points_2);
P_mat_2 = zeros(n_all, 20);
for i = 1:n_all
    w = exp(-EPSILON * distMat_2(i,:) / 2.0);
    P_mat_2(i,:) = w / sum(w);
end
cost_2 = loss_mat_2_all / n_all;
loss_EM_2 = sum(sum(cost_2 .* P_mat_2));
prior_2 = ones(1, n_all) / n_all;
BR_2 = perturbation_cal_rmp(P_mat_2, prior_2, loss_mat_2_all);
fprintf('[eps=%.1f] Scenario 2 (random 20 nodes):\n', EPSILON);
fprintf('  loss_EM=%.1f, loss_EMBR=%.1f, ratio=%.2fx\n', loss_EM_2, BR_2, loss_EM_2/BR_2);

%% === Key insight: P_matrix entropy (how concentrated are the probabilities?) ===
fprintf('\n--- P_matrix row entropy (how "peaked" is each row?) ---\n');

entropy_1 = zeros(n_all, 1);
for i = 1:n_all
    p = P_mat_1(i,:);
    p = p(p > 0);
    entropy_1(i) = -sum(p .* log(p));
end
entropy_2 = zeros(n_all, 1);
for i = 1:n_all
    p = P_mat_2(i,:);
    p = p(p > 0);
    entropy_2(i) = -sum(p .* log(p));
end
max_entropy = log(20);
fprintf('Scenario 1 (first 20): mean entropy=%.3f, max_possible=%.3f (%.1f%% of max)\n', ...
    mean(entropy_1), max_entropy, 100*mean(entropy_1)/max_entropy);
fprintf('Scenario 2 (random 20): mean entropy=%.3f, max_possible=%.3f (%.1f%% of max)\n', ...
    mean(entropy_2), max_entropy, 100*mean(entropy_2)/max_entropy);

%% === What per_point does EM mostly choose in each scenario? ===
fprintf('\n--- Which per_point does EM predominantly select? ---\n');
[~, most_likely_1] = max(P_mat_1, [], 2);
[~, most_likely_2] = max(P_mat_2, [], 2);
fprintf('Scenario 1: unique dominant per_points: %d (out of 20)\n', numel(unique(most_likely_1)));
fprintf('Scenario 2: unique dominant per_points: %d (out of 20)\n', numel(unique(most_likely_2)));

counts_1 = histcounts(most_likely_1, 1:21);
counts_2 = histcounts(most_likely_2, 1:21);
fprintf('Scenario 1 dominant counts (top 5): ');
[~, order1] = sort(counts_1, 'descend');
fprintf('%d(x%d) ', [order1(1:5); counts_1(order1(1:5))]); fprintf('\n');
fprintf('Scenario 2 dominant counts (top 5): ');
[~, order2] = sort(counts_2, 'descend');
fprintf('%d(x%d) ', [order2(1:5); counts_2(order2(1:5))]); fprintf('\n');

%% === Correlation: does geographic proximity predict path-length proximity? ===
fprintf('\n--- Correlation: geographic dist vs path-length loss ---\n');
% For each allPoint, which per_point is geographically nearest vs path-length nearest?
[~, geo_nearest_1] = min(distMat_1, [], 2);
[~, loss_nearest_1] = min(loss_mat_1_all, [], 2);
agree_1 = sum(geo_nearest_1 == loss_nearest_1) / n_all;

[~, geo_nearest_2] = min(distMat_2, [], 2);
[~, loss_nearest_2] = min(loss_mat_2_all, [], 2);
agree_2 = sum(geo_nearest_2 == loss_nearest_2) / n_all;

fprintf('Scenario 1: geo-nearest == loss-nearest agreement: %.1f%%\n', 100*agree_1);
fprintf('Scenario 2: geo-nearest == loss-nearest agreement: %.1f%%\n', 100*agree_2);

%% === Also compare with main_2norm.m style: both true+perturbed from road network ===
fprintf('\n--- Scenario 3: BOTH true and perturbed from road network (main_2norm.m style) ---\n');
% True locations = all real road nodes, not allPoints
true_points = [col_latitude, col_longitude];
n_true = size(true_points, 1);

loss_mat_3 = zeros(n_true, 20);
for ii = 1:n_true
    for jj = 1:20
        loss_mat_3(ii, jj) = abs(pathLength(1, ii) - pathLength(1, perturbed_indices_2(jj)));
    end
end

distMat_3 = pdist2(true_points, per_points_2);
P_mat_3 = zeros(n_true, 20);
for i = 1:n_true
    w = exp(-EPSILON * distMat_3(i,:) / 2.0);
    P_mat_3(i,:) = w / sum(w);
end
cost_3 = loss_mat_3 / n_true;
loss_EM_3 = sum(sum(cost_3 .* P_mat_3));
prior_3 = ones(1, n_true) / n_true;
BR_3 = perturbation_cal_rmp(P_mat_3, prior_3, loss_mat_3);
fprintf('[eps=%.1f] Scenario 3 (main_2norm.m style):\n', EPSILON);
fprintf('  loss_EM=%.1f, loss_EMBR=%.1f, ratio=%.2fx\n', loss_EM_3, BR_3, loss_EM_3/BR_3);

entropy_3 = zeros(n_true, 1);
for i = 1:n_true
    p = P_mat_3(i,:); p = p(p>0);
    entropy_3(i) = -sum(p .* log(p));
end
fprintf('  P_matrix mean entropy: %.3f (%.1f%% of max %.3f)\n', ...
    mean(entropy_3), 100*mean(entropy_3)/max_entropy, max_entropy);

[~, geo_nearest_3] = min(distMat_3, [], 2);
[~, loss_nearest_3] = min(loss_mat_3, [], 2);
agree_3 = sum(geo_nearest_3 == loss_nearest_3) / n_true;
fprintf('  geo-nearest == loss-nearest agreement: %.1f%%\n', 100*agree_3);

fprintf('\n=== Summary ===\n');
fprintf('Scenario 1 (first20 + allPoints):  EM=%.1f, EMBR=%.1f, ratio=%.2fx\n', loss_EM_1, BR_1, loss_EM_1/BR_1);
fprintf('Scenario 2 (random20 + allPoints): EM=%.1f, EMBR=%.1f, ratio=%.2fx\n', loss_EM_2, BR_2, loss_EM_2/BR_2);
fprintf('Scenario 3 (random20 + roadnodes): EM=%.1f, EMBR=%.1f, ratio=%.2fx\n', loss_EM_3, BR_3, loss_EM_3/BR_3);
fprintf('\n=== Done ===\n');
