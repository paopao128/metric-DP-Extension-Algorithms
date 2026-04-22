%% Diagnose: why is loss_aipo_instance still higher than EM?
addpath('./functions/haversine'); addpath('./functions'); addpath('./functions/benchmarks');
parameters;
city_list = {'rome'}; city_idx = 1; city = city_list{city_idx};
node_file = sprintf('./datasets/%s/nodes.csv', city);
opts = detectImportOptions(node_file); opts = setvartype(opts, 'osmid', 'int64');
df_nodes = readtable(node_file, opts);
col_longitude_orig = table2array(df_nodes(:,'x')); col_latitude_orig = table2array(df_nodes(:,'y'));
max_longitude=max(col_longitude_orig); min_longitude=min(col_longitude_orig); mid_longitude=(max_longitude+min_longitude)/2; LONGITUDE_SIZE=max_longitude-min_longitude;
max_latitude=max(col_latitude_orig); min_latitude=min(col_latitude_orig); mid_latitude=(max_latitude+min_latitude)/2; LATITUDE_SIZE=max_latitude-min_latitude;
lon_range=[mid_longitude-LONGITUDE_SIZE/SCALE, mid_longitude+LONGITUDE_SIZE/SCALE];
lat_range=[mid_latitude-LATITUDE_SIZE/SCALE, mid_latitude+LATITUDE_SIZE/SCALE];
selected_indices=filter_coords_by_range(col_longitude_orig,col_latitude_orig,lon_range,lat_range);
col_longitude=col_longitude_orig(selected_indices,:); col_latitude=col_latitude_orig(selected_indices,:);
[col_longitude, col_latitude]=lonlat_to_xy(col_longitude,col_latitude,mid_longitude,mid_latitude);
NR_REAL_LOC=size(col_longitude,1);
rng(42);
perturbed_indices=randi([1,NR_REAL_LOC],1,NR_PER_LOC);
perturbed_xy=[col_latitude(perturbed_indices',1), col_longitude(perturbed_indices',1)];
loss_matrix=pdist2([col_latitude,col_longitude],perturbed_xy);
loss_matrix_max=min(loss_matrix,[],2);
[adjMatrix,distanceMatrix,neighborMatrix,cornerPoints_ori,~,~,~,corner_weights]=uniform_anchor(col_latitude,col_longitude,loss_matrix_max,cell_size(1,city_idx));
close all;

EPSILON = 0.5; epsilon_idx_1 = 2;
epsilon_1 = EPSILON*epsilon_idx_1/NR_EPSILON_INTERVAL;
epsilon_2 = EPSILON*sqrt(1-(epsilon_idx_1/NR_EPSILON_INTERVAL)^2);

loss_matrix_selected = loss_matrix;
corner_weights_selected_ori = corner_weights;
c_approx = corner_weights_selected_ori' * loss_matrix_selected;
[z_anchor, ~] = perturbation_cal_apo(c_approx, corner_weights_selected_ori, distanceMatrix, neighborMatrix, epsilon_1/2, epsilon_2/2);

%% ===== TEST 1: Check logconv_interp formula with sparse W =====
fprintf('===== TEST 1: logconv_interp formula with sparse vs full W =====\n');
cornerPoints = cornerPoints_ori; allPoints = cornerPoints_ori;
cornerPoints = allPoints; interpolation_extension_k;
W_sparse = W;
W_full = full(W);
% Run logconv_interp with sparse W
[x_sparse, loss_sparse] = logconv_interp(z_anchor, W_sparse, pdist2(allPoints, perturbed_xy));
% Run logconv_interp with full W
[x_full,   loss_full]   = logconv_interp(z_anchor, W_full,   pdist2(allPoints, perturbed_xy));
fprintf('loss with sparse W: %.6f\n', loss_sparse);
fprintf('loss with full   W: %.6f\n', loss_full);
fprintf('Max diff in x_opt:  %.2e\n', max(abs(x_sparse(:)-x_full(:))));

%% Check if rows sum to 1
rowsums_sparse = sum(x_sparse, 2);
rowsums_full   = sum(x_full,   2);
fprintf('Row sums (sparse W): min=%.4f, max=%.4f, mean=%.4f\n', min(rowsums_sparse), max(rowsums_sparse), mean(rowsums_sparse));
fprintf('Row sums (full   W): min=%.4f, max=%.4f, mean=%.4f\n', min(rowsums_full), max(rowsums_full), mean(rowsums_full));
fprintf('NaN rows (sparse): %d, (full): %d\n', sum(isnan(rowsums_sparse)), sum(isnan(rowsums_full)));

%% ===== TEST 2: Check the .^ dimension behavior for a single row =====
fprintf('\n===== TEST 2: Dimension check of .^ in logconv_interp =====\n');
i_test = 1; k_test = 1;
vec_a = z_anchor(:, k_test)';   % 1×J (dense)
vec_b = W_sparse(i_test, :)';   % J×1 (sparse)
vec_b_full = W_full(i_test, :)'; % J×1 (full)
fprintf('z_anchor(:,k)'': size = %s (dense)\n', mat2str(size(vec_a)));
fprintf('W_sparse(i,:)'': size = %s (sparse)\n', mat2str(size(vec_b)));
fprintf('W_full(i,:)'':   size = %s (full)\n',   mat2str(size(vec_b_full)));

result_sparse = vec_a .^ vec_b;
result_full   = vec_a .^ vec_b_full;
fprintf('(1×J).^(J×1) sparse → size = %s\n', mat2str(size(result_sparse)));
fprintf('(1×J).^(J×1) full   → size = %s\n', mat2str(size(result_full)));
fprintf('prod result (sparse): size=%s, val=%.6f\n', mat2str(size(prod(result_sparse))), prod(prod(result_sparse)));
fprintf('prod result (full):   size=%s, val=%.6f\n', mat2str(size(prod(result_full))),   prod(prod(result_full)));

%% ===== TEST 3: Compare EM vs log-convex interpolation at each layer =====
fprintf('\n===== TEST 3: EM vs logconv_interp at each grid layer (epsilon=0.5) =====\n');
allPoints_cur = cornerPoints_ori;
z_cur = z_anchor;
for layer = 1:3
    cornerPoints = allPoints_cur;
    interpolation_extension_k;
    distMat = pdist2(allPoints, perturbed_xy);

    % EM loss at this layer
    P_em = exp(-EPSILON * distMat / 2);
    P_em = P_em ./ sum(P_em, 2);
    loss_em_layer = sum(sum(P_em .* distMat)) / size(distMat, 1);

    % logconv_interp loss at this layer (full W)
    [z_ext, loss_logconv] = logconv_interp(z_cur, full(W), distMat);

    % bilinear interpolation of z_anchor for comparison
    z_bilinear = full(W) * z_cur;
    z_bilinear = z_bilinear ./ sum(z_bilinear, 2);
    loss_bilinear = sum(sum(z_bilinear .* distMat)) / size(distMat, 1);

    fprintf('Layer %d (%d pts): EM=%.4f, logconv=%.4f, bilinear=%.4f\n', ...
        layer, size(allPoints,1), loss_em_layer, loss_logconv, loss_bilinear);

    allPoints_cur = allPoints;
    z_cur = z_ext;
end

%% ===== TEST 4: Weighted loss at REAL locations vs grid points =====
fprintf('\n===== TEST 4: Loss at real locations vs grid points =====\n');
% After 3 rounds
allPoints_cur = cornerPoints_ori; z_cur = z_anchor;
for layer = 1:3
    cornerPoints = allPoints_cur;
    interpolation_extension_k;
    distMat_fine = pdist2(allPoints, perturbed_xy);
    [z_ext, ~] = logconv_interp(z_cur, full(W), distMat_fine);
    allPoints_cur = allPoints;
    z_cur = z_ext;
end
% z_cur is now fine-grid distribution (nFine × NR_PER_LOC)
% For each real location, find nearest fine grid point
[~, nearest_fine] = min(pdist2([col_latitude, col_longitude], allPoints_cur), [], 2);
z_at_real = z_cur(nearest_fine, :);  % assign each real loc the nearest fine grid distribution
loss_at_real = sum(sum(z_at_real .* loss_matrix)) / NR_REAL_LOC;
loss_at_fine  = sum(sum(z_cur .* distMat_fine)) / size(allPoints_cur,1);
fprintf('Loss averaged over fine grid pts: %.4f\n', loss_at_fine);
fprintf('Loss averaged over REAL locs:     %.4f\n', loss_at_real);

% EM at real locations directly
P_em_real = exp(-EPSILON * loss_matrix / 2);
P_em_real = P_em_real ./ sum(P_em_real, 2);
loss_em_real = sum(sum(P_em_real .* loss_matrix)) / NR_REAL_LOC;
fprintf('EM loss at REAL locs directly:    %.4f\n', loss_em_real);
fprintf('EM loss at fine grid (layer 3):   %.4f\n', ...
    sum(sum(exp(-EPSILON*distMat_fine/2) ./ sum(exp(-EPSILON*distMat_fine/2),2) .* distMat_fine)) / size(distMat_fine,1));
