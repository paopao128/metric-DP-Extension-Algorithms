%% Final comparison: AIPO vs EM at each interpolation layer
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

fprintf('NR_REAL_LOC=%d, NR_ANCHOR=%d, k=%d in interpolation_extension_k\n', ...
    NR_REAL_LOC, size(cornerPoints_ori,1), 3);  % k=3 now

%% For each epsilon, find best LP solution and compare with EM at each layer
fprintf('\n%-8s %-8s %-10s %-10s %-12s %-10s\n', 'EPSILON', 'layer', 'nPts', 'EM', 'AIPO(best)', 'AIPO/EM');
for epsilon_idx = 1:EPSILON_MAX
    EPSILON = 0.5 * epsilon_idx;
    c_approx = corner_weights' * loss_matrix;

    % Find best LP solution over epsilon splits
    best_loss_layer3 = inf;
    best_z = [];
    for epsilon_idx_1 = 1:NR_EPSILON_INTERVAL-1
        epsilon_1 = EPSILON * epsilon_idx_1 / NR_EPSILON_INTERVAL;
        epsilon_2 = EPSILON * sqrt(1 - (epsilon_idx_1/NR_EPSILON_INTERVAL)^2);
        [z_anc, ~] = perturbation_cal_apo(c_approx, corner_weights, distanceMatrix, neighborMatrix, epsilon_1/2, epsilon_2/2);

        % Run 3 interpolation rounds and get loss at layer 3
        allPts = cornerPoints_ori; z_cur = z_anc;
        for layer = 1:3
            cornerPoints = allPts; interpolation_extension_k;
            dist_fine = pdist2(allPoints, perturbed_xy);
            [z_ext, loss_l] = logconv_interp(z_cur, W, dist_fine);
            allPts = allPoints; z_cur = z_ext;
        end
        if loss_l < best_loss_layer3
            best_loss_layer3 = loss_l;
            best_z_anc = z_anc;
        end
    end

    % Compute EM loss at each layer (compare same layer)
    allPts = cornerPoints_ori; z_cur = best_z_anc;
    for layer = 1:3
        cornerPoints = allPts; interpolation_extension_k;
        dist_fine = pdist2(allPoints, perturbed_xy);

        % EM at this layer
        P_em = exp(-EPSILON * dist_fine / 2);
        P_em = P_em ./ sum(P_em, 2);
        loss_em = sum(sum(P_em .* dist_fine)) / size(dist_fine, 1);

        % AIPO at this layer
        [z_ext, loss_aipo_layer] = logconv_interp(z_cur, W, dist_fine);

        if layer == 3
            fprintf('%-8.1f %-8d %-10d %-10.4f %-12.4f %-10.3f\n', ...
                EPSILON, layer, size(allPoints,1), loss_em, loss_aipo_layer, loss_aipo_layer/loss_em);
        end
        allPts = allPoints; z_cur = z_ext;
    end
end

%% Check: does LP solution AT ANCHOR POINTS beat EM at anchor points?
fprintf('\n--- At ORIGINAL ANCHOR LEVEL (before interpolation) ---\n');
fprintf('%-8s %-10s %-10s %-10s\n', 'EPSILON', 'EM@anchor', 'AIPO@anchor', 'ratio');
for epsilon_idx = 1:EPSILON_MAX
    EPSILON = 0.5 * epsilon_idx;
    c_approx = corner_weights' * loss_matrix;

    dist_anc = pdist2(cornerPoints_ori, perturbed_xy);
    P_em_anc = exp(-EPSILON * dist_anc / 2);
    P_em_anc = P_em_anc ./ sum(P_em_anc, 2);
    loss_em_anc = sum(sum(P_em_anc .* dist_anc)) / size(dist_anc, 1);

    best_loss_anc = inf;
    for epsilon_idx_1 = 1:NR_EPSILON_INTERVAL-1
        epsilon_1 = EPSILON * epsilon_idx_1 / NR_EPSILON_INTERVAL;
        epsilon_2 = EPSILON * sqrt(1 - (epsilon_idx_1/NR_EPSILON_INTERVAL)^2);
        [z_anc, fval] = perturbation_cal_apo(c_approx, corner_weights, distanceMatrix, neighborMatrix, epsilon_1/2, epsilon_2/2);
        loss_anc = sum(sum(z_anc .* dist_anc)) / size(dist_anc, 1);
        if loss_anc < best_loss_anc
            best_loss_anc = loss_anc;
        end
    end
    fprintf('%-8.1f %-10.4f %-10.4f %-10.3f\n', EPSILON, loss_em_anc, best_loss_anc, best_loss_anc/loss_em_anc);
end
