%% Header
addpath('./functions/haversine');
addpath('./functions');
addpath('./functions/benchmarks');

fprintf('------------------- Environment settings --------------------- \n\n');

parameters;
city_list = {'rome', 'nyc', 'london'};

l_p = 2; % l_2 norm

for city_idx = 1:3 % length(city_list)
    city = city_list{city_idx};
    fprintf('\n------------- Processing city: %s -------------\n', city);

    %% Read map information
    node_file = sprintf('./datasets/%s/nodes.mat', city);
    load(node_file, 'col_longitude_orig', 'col_latitude_orig');

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

%% ------------------------ Start running the simulation here ------------------------------
    for test_idx = 1:1:NR_TEST
        perturbed_indices = randi([1, NR_REAL_LOC], 1, NR_PER_LOC);
        per_points = [col_latitude(perturbed_indices', 1), col_longitude(perturbed_indices', 1)];
        loss_matrix_max = min(pdist2([col_latitude, col_longitude], per_points), [], 2);
        [~, ~, ~, cornerPoints_ori, ~, ~, ~, ~] = uniform_anchor(col_latitude, col_longitude, loss_matrix_max, cell_size(1, city_idx));
        close all;

        allPoints = cornerPoints_ori;
        for layer_id = 1:1:3
            cornerPoints = allPoints;
            interpolation_extension_k;
            distMat = pdist2(allPoints, per_points);

            for epsilon_idx = 1:1:EPSILON_MAX
                EPSILON = 0.5 * epsilon_idx;

                t_lap_start = tic;
                [~, loss_lap] = perturbation_cal_laplace( ...
                    allPoints(:,1), allPoints(:,2), ...
                    per_points(:,1), per_points(:,2), ...
                    distMat, EPSILON);
                time_lap(epsilon_idx + 3*layer_id - 3) = toc(t_lap_start);

                loss_laplace(epsilon_idx + 3*layer_id - 3) = loss_lap;
            end
        end
        all_laplace(test_idx, :) = loss_laplace;
        all_time_lap(test_idx, :) = time_lap;
        fprintf('Test index %d done.\n', test_idx);
    end
    mean_laplace = mean(all_laplace, 1);
    std_laplace  = std(all_laplace,  0, 1);
    parts = cell(1, 9);
    for k = 1:9
        parts{k} = sprintf('%.2f%s%.2f', mean_laplace(k), char(177), std_laplace(k));
    end
    fprintf('[%s] %s\n', city, strjoin(parts, ' & '));

    mean_time_lap = mean(all_time_lap, 1);
    std_time_lap  = std(all_time_lap, 0, 1);
    parts_t = cell(1, 9);
    for k = 1:9
        parts_t{k} = sprintf('%.4f%s%.4f', mean_time_lap(k), char(177), std_time_lap(k));
    end
    fprintf('laplace [%s] time: %s\n', city, strjoin(parts_t, ' & '));
    fid = fopen('results_time.txt', 'a');
    fprintf(fid, 'laplace [%s]: %s\n', city, strjoin(parts_t, ' & '));
    fclose(fid);
end
