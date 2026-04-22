%% Header
addpath('./functions/haversine');
addpath('./functions');
addpath('./functions/benchmarks');

fprintf('------------------- Environment settings --------------------- \n\n');

parameters;
city_list = {'rome', 'nyc', 'london'};

l_p = 2; % l_2 norm

for city_idx = 3:3
    city = city_list{city_idx};
    fprintf('\n------------- Processing city: %s -------------\n', city);

    %% Read map information
    node_file = sprintf('./datasets/%s/nodes.csv', city);
    edge_file  = sprintf('./datasets/%s/edges.csv', city);

    opts = detectImportOptions(node_file);
    opts = setvartype(opts, 'osmid', 'int64');
    df_nodes = readtable(node_file, opts);
    df_edges = readtable(edge_file);

    col_longitude_orig = table2array(df_nodes(:, 'x'));
    col_latitude_orig  = table2array(df_nodes(:, 'y'));
    col_osmid          = table2array(df_nodes(:, 'osmid'));

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
    prior = ones(1, NR_REAL_LOC) / NR_REAL_LOC;

    % Per-city accumulators: EPSILON_MAX x NR_TEST
    loss_lp_all      = zeros(EPSILON_MAX, NR_TEST);
    violation_lp_all = zeros(EPSILON_MAX, NR_TEST);
    time_all_tests   = zeros(NR_TEST, EPSILON_MAX);

%% ------------------------ Start running the simulation here ------------------------------
    for test_idx = 1:NR_TEST
        perturbed_indices = randi([1, NR_REAL_LOC], 1, NR_PER_LOC);
        perturbed_xy      = [col_latitude(perturbed_indices', 1), col_longitude(perturbed_indices', 1)];
        loss_matrix       = pdist2([col_latitude, col_longitude], perturbed_xy);

        samples_viocheck = randperm(NR_REAL_LOC, SAMPLE_SIZE_PPR);
        euclidean_distance_matrix_viocheck = distance_matrix( ...
            col_latitude(samples_viocheck), col_longitude(samples_viocheck), l_p);

        for epsilon_idx = 1:EPSILON_MAX
            EPSILON = 0.5 * epsilon_idx;   % epsilon: 0.5, 1.0, 1.5

            selected_longitudes  = col_longitude;
            selected_latitudes   = col_latitude;
            loss_matrix_selected = loss_matrix;

            % Adaptive grid size: use finer grid for larger epsilon
            if epsilon_idx >= 3
                GRID_SIZE_LP_INST = GRID_SIZE_LP;   % full grid (18x18 by default)
            else
                GRID_SIZE_LP_INST = 12;             % coarser grid for small epsilon
            end

            %% LP computation (timed from partition_grid through convert)
            t_start = tic;

            [grid_utility_loss, grid_distances, grid_prior, ~, ~] = partition_grid( ...
                selected_longitudes, selected_latitudes, loss_matrix_selected, prior, ...
                GRID_SIZE_LP_INST, GRID_SIZE_LP_INST, ...
                col_longitude(perturbed_indices), col_latitude(perturbed_indices), l_p);

            [z_lp, loss_lp_val] = perturbation_cal_lp( ...
                grid_utility_loss, grid_distances, grid_prior, EPSILON);

            z_lp_fine = convert_grid_to_fine_perturbation( ...
                z_lp, col_longitude, col_latitude, GRID_SIZE_LP_INST, GRID_SIZE_LP_INST);

            time_lp_ep(epsilon_idx) = toc(t_start);

            loss_lp_all(epsilon_idx, test_idx) = loss_lp_val;

            [violation_lp_all(epsilon_idx, test_idx), ~] = compute_mDP_violation( ...
                z_lp_fine(samples_viocheck, :), euclidean_distance_matrix_viocheck, EPSILON);
        end

        time_all_tests(test_idx, :) = time_lp_ep;
        fprintf('Test index %d done.\n', test_idx);
    end

    %% Output: 9 values per metric
    % LP has no layer structure; replicate 3 epsilon values across 3 "layer" slots
    % so positions match (epsilon_idx + 3*(layer_id-1)) used by other methods.

    % --- Loss ---
    mean_loss = mean(loss_lp_all, 2);   % EPSILON_MAX x 1
    std_loss  = std(loss_lp_all,  0, 2);
    all_mean_l = repmat(mean_loss, 1, 3);
    all_std_l  = repmat(std_loss,  1, 3);
    parts_l = cell(1, 9);
    for k = 1:9
        parts_l{k} = sprintf('%.2f%s%.2f', all_mean_l(k), char(177), all_std_l(k));
    end
    fprintf('[%s] loss: %s\n', city, strjoin(parts_l, ' & '));

    % --- Computation time ---
    mean_time = mean(time_all_tests, 1);   % 1 x EPSILON_MAX
    std_time  = std(time_all_tests,  0, 1);
    all_mean_t = repmat(mean_time(:), 1, 3);
    all_std_t  = repmat(std_time(:),  1, 3);
    parts_t = cell(1, 9);
    for k = 1:9
        parts_t{k} = sprintf('%.2f%s%.2f', all_mean_t(k), char(177), all_std_t(k));
    end
    fprintf('[%s] time: %s\n', city, strjoin(parts_t, ' & '));

    % --- Violation rate ---
    mean_vio = mean(violation_lp_all, 2);
    std_vio  = std(violation_lp_all,  0, 2);
    all_mean_v = repmat(mean_vio, 1, 3);
    all_std_v  = repmat(std_vio,  1, 3);
    parts_v = cell(1, 9);
    for k = 1:9
        parts_v{k} = sprintf('%.4f%s%.4f', all_mean_v(k), char(177), all_std_v(k));
    end
    fprintf('[%s] violation: %s\n', city, strjoin(parts_v, ' & '));

    % --- Append to results file ---
    fid = fopen('results_LP_A.txt', 'a');
    fprintf(fid, 'loss [%s]: %s\n',      city, strjoin(parts_l, ' & '));
    fprintf(fid, 'time [%s]: %s\n',      city, strjoin(parts_t, ' & '));
    fprintf(fid, 'violation [%s]: %s\n', city, strjoin(parts_v, ' & '));
    fclose(fid);
end
