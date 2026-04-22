%% save_point_data.m
% Save original locations and obfuscated locations (tree_MWE algorithm)
% for all 3 cities x 3 layers x 5 tests.
% Uses epsilon = 1.0 (epsilon_idx = 2) with optimal budget split.
% Obfuscated location = expected output = allPoints' * z_extension.
% Output: points/[city]_layer[k].csv with columns [test_id, orig_lat, orig_lon, obf_lat, obf_lon]
% Coordinates are in km (xy after lonlat_to_xy conversion).

addpath('./functions/haversine');
addpath('./functions');
addpath('./functions/benchmarks');

parameters;
city_list = {'rome', 'nyc', 'london'};

EPSILON_SAVE = 1.0;  % fixed epsilon for visualization

mkdir('points');

for city_idx = 1:3
    city = city_list{city_idx};
    fprintf('\n--- Processing city: %s ---\n', city);

    node_file = sprintf('./datasets/%s/nodes.csv', city);
    edge_file  = sprintf('./datasets/%s/edges.csv',  city);

    opts = detectImportOptions(node_file);
    opts = setvartype(opts, 'osmid', 'int64');
    df_nodes = readtable(node_file, opts);
    df_edges = readtable(edge_file);

    col_longitude_orig = table2array(df_nodes(:, 'x'));
    col_latitude_orig  = table2array(df_nodes(:, 'y'));

    max_longitude = max(col_longitude_orig);
    min_longitude = min(col_longitude_orig);
    mid_longitude = (max_longitude + min_longitude) / 2;
    LONGITUDE_SIZE = max_longitude - min_longitude;

    max_latitude = max(col_latitude_orig);
    min_latitude = min(col_latitude_orig);
    mid_latitude = (max_latitude + min_latitude) / 2;
    LATITUDE_SIZE = max_latitude - min_latitude;

    lon_range = [mid_longitude - LONGITUDE_SIZE/SCALE, mid_longitude + LONGITUDE_SIZE/SCALE];
    lat_range = [mid_latitude  - LATITUDE_SIZE/SCALE,  mid_latitude  + LATITUDE_SIZE/SCALE];

    selected_indices = filter_coords_by_range(col_longitude_orig, col_latitude_orig, lon_range, lat_range);
    col_longitude = col_longitude_orig(selected_indices, :);
    col_latitude  = col_latitude_orig(selected_indices, :);
    [col_longitude, col_latitude] = lonlat_to_xy(col_longitude, col_latitude, mid_longitude, mid_latitude);

    NR_REAL_LOC = size(col_longitude, 1);

    % Pre-allocate storage: each layer gets NR_TEST*NR_PER_LOC rows
    data_layer = cell(1, 3);
    for k = 1:3
        data_layer{k} = zeros(NR_TEST * NR_PER_LOC, 5);  % [test_id, orig_lat, orig_lon, obf_lat, obf_lon]
    end

    for test_idx = 1:NR_TEST
        perturbed_indices = randi([1, NR_REAL_LOC], 1, NR_PER_LOC);
        perturbed_xy = [col_latitude(perturbed_indices', 1), col_longitude(perturbed_indices', 1)];
        loss_matrix = pdist2([col_latitude, col_longitude], perturbed_xy);
        loss_matrix_max = min(loss_matrix, [], 2);
        [adjMatrix, distanceMatrix, neighborMatrix, cornerPoints_ori, ~, ~, ~, corner_weights] = ...
            uniform_anchor(col_latitude, col_longitude, loss_matrix_max, cell_size(1, city_idx));
        close all;

        loss_matrix_selected     = loss_matrix;
        corner_weights_selected_ori = corner_weights;
        anchor_prior = full(corner_weights_selected_ori)' * ones(NR_REAL_LOC, 1) / NR_REAL_LOC;
        c_approx = corner_weights_selected_ori' * loss_matrix_selected;

        % Search best budget split for EPSILON_SAVE
        best_loss = inf;
        best_z      = cell(1, 3);
        best_pts    = cell(1, 3);

        % Compute allPoints per layer once (independent of budget split)
        allPoints_by_layer = cell(1, 3);
        tmp_allPoints = cornerPoints_ori;
        for interp_id = 1:3
            cornerPoints = tmp_allPoints;
            interpolation_extension_k;          % produces new allPoints and W
            allPoints_by_layer{interp_id} = allPoints;
            tmp_allPoints = allPoints;
        end

        for epsilon_idx_1 = 1:NR_EPSILON_INTERVAL-1
            epsilon_1 = EPSILON_SAVE * epsilon_idx_1 / NR_EPSILON_INTERVAL;
            epsilon_2 = EPSILON_SAVE * sqrt(1 - (epsilon_idx_1/NR_EPSILON_INTERVAL)^2);
            corner_weights_selected = corner_weights_selected_ori;
            cornerPoints = cornerPoints_ori;

            [z_anchor_instance, ~] = perturbation_cal_apo(c_approx, corner_weights_selected, ...
                distanceMatrix, neighborMatrix, epsilon_1/2, epsilon_2/2);

            allPoints = cornerPoints_ori;
            current_prior = anchor_prior;
            z_split = cell(1, 3);

            for interp_id = 1:3
                cornerPoints = allPoints;
                interpolation_extension_k;     % produces new allPoints and W
                corner_weights_selected = W;
                current_prior = full(W) * current_prior;
                current_prior = current_prior / sum(current_prior);
                loss_matrix_selected = pdist2(allPoints, perturbed_xy);
                if interp_id > 1
                    z_anchor_instance = z_extension;
                end
                [z_extension, loss_val] = MWE_cell_BFS(cornerPoints, z_anchor_instance, ...
                    allPoints, 0.5, loss_matrix_selected, current_prior);
                z_split{interp_id} = z_extension;
            end

            if loss_val < best_loss
                best_loss = loss_val;
                best_z = z_split;
                % allPoints_by_layer already fixed; just confirm
            end
        end

        % Compute expected obfuscated locations for each layer
        row_start = (test_idx - 1) * NR_PER_LOC + 1;
        row_end   = test_idx * NR_PER_LOC;

        for layer_id = 1:3
            pts = allPoints_by_layer{layer_id};   % (N_pts x 2)
            z   = best_z{layer_id};               % (N_pts x NR_PER_LOC)
            obf = (pts' * z)';                    % (NR_PER_LOC x 2): expected [lat, lon]

            data_layer{layer_id}(row_start:row_end, :) = ...
                [repmat(test_idx, NR_PER_LOC, 1), perturbed_xy, obf];
        end

        fprintf('  Test %d/%d done.\n', test_idx, NR_TEST);
    end

    % Write CSV files: one per (city, layer)
    for layer_id = 1:3
        fname = sprintf('points/%s_layer%d.csv', city, layer_id);
        header = {'test_id', 'orig_lat_km', 'orig_lon_km', 'obf_lat_km', 'obf_lon_km'};
        T = array2table(data_layer{layer_id}, 'VariableNames', header);
        writetable(T, fname);
        fprintf('Saved: %s\n', fname);
    end
end

fprintf('\nDone. Files saved in points/ folder.\n');
