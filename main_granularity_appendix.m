%% Header
addpath('./functions/haversine');  
addpath('./functions');
addpath('./functions/benchmarks');

l_p = 2; 

fprintf('------------------- Environment settings --------------------- \n\n');
rng("default");

parameters; 

city_list = {'rome', 'nyc', 'london'};

for city_idx = 1:length(city_list)
    city = city_list{city_idx};
    fprintf('\n------------- Processing city: %s -------------\n', city);

    %% Read map information
    node_file = sprintf('./datasets/%s/nodes.csv', city);
    edge_file = sprintf('./datasets/%s/edges.csv', city);

    opts = detectImportOptions(node_file);
    opts = setvartype(opts, 'osmid', 'int64');
    df_nodes = readtable(node_file, opts);
    df_edges = readtable(edge_file);

    col_longitude_orig = table2array(df_nodes(:, 'x'));
    col_latitude_orig = table2array(df_nodes(:, 'y'));
    col_osmid = table2array(df_nodes(:, 'osmid'));
    NR_LOC = size(df_nodes, 1);

    % Define the range of longitude and latitude
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
    
    % Select the coordinates with the given range
    selected_indices = filter_coords_by_range(col_longitude_orig, col_latitude_orig, lon_range, lat_range); 

    col_longitude = col_longitude_orig(selected_indices, :); 
    col_latitude = col_latitude_orig(selected_indices, :); 

    % Convert the longitude and latitude coordinates to xy coordinates  
    [col_longitude, col_latitude] = lonlat_to_xy(col_longitude, col_latitude, mid_longitude, mid_latitude);
    LONGITUDE_SIZE_XY = max(col_longitude) - min(col_longitude); 

    NR_REAL_LOC = size(col_longitude, 1); 
    
    % Read the utiltiy loss matrix here
    path_file = sprintf('./intermediate/%s/loss_matrix_orig.mat', city);

    load(path_file);

    % Initialize metrics
    time_aipo = 0; 
    loss_aipo = 0; 
    violation_aipo = 0; 

    nr_anchor = 0; 

    for test_idx = 1:1:NR_TEST
        samples_viocheck = randperm(NR_LOC, NR_VIO_SAMPLE);

        euclidean_distance_matrix_viocheck = distance_matrix(col_latitude(samples_viocheck), col_longitude(samples_viocheck), l_p); 
        for grid_size_index = 1:1:15
            perturbed_indices = randi([1, NR_LOC], 1, NR_PER_LOC);
            perturbed_longitudes = col_longitude(perturbed_indices);
            perturbed_latitudes = col_latitude(perturbed_indices);

            grid_size = grid_size_index*1; 
            perturbed_indices = randi([1, NR_LOC], 1, NR_PER_LOC);
            
            loss_matrix = loss_matrix_orig;
            loss_matrix_max = min(loss_matrix, [], 2); 
            
            [adjMatrix, distanceMatrix, neighborMatrix, cornerPoints, squares, lambda_x, lambda_y, corner_weights] = uniform_anchor(col_latitude, col_longitude, loss_matrix_max, LONGITUDE_SIZE_XY/grid_size);
            close all; 
            nr_anchor(grid_size_index, test_idx) = size(cornerPoints, 1); 

            N = NR_LOC; 
            selected_indices = 1:NR_LOC; 

            selected_longitudes = col_longitude(selected_indices);
            selected_latitudes = col_latitude(selected_indices);
            selected_osmids = col_osmid(selected_indices);
            loss_matrix_selected = loss_matrix(selected_indices, :); 
            corner_weights_selected = corner_weights(selected_indices, :);

            %% Interporlation
            c_approx = corner_weights_selected'*loss_matrix_selected; 
            tic
            loss_aipo(grid_size_index, test_idx) = 999999999;
            for epsilon_idx_1 = 1:1:NR_EPSILON_INTERVAL-1                   % Discretize epsilon_1 and try it one by one
                epsilon_1 = EPSILON*epsilon_idx_1/NR_EPSILON_INTERVAL; 
                epsilon_2 = EPSILON*sqrt(1-(epsilon_idx_1/NR_EPSILON_INTERVAL)^2);           % Given epsilon 1, calculate the corresponding epsilon_2

                % Calculate the perturbation matrix for anchor records using "perturbation_cal_apo"
                z_anchor_instance = perturbation_cal_apo(c_approx, corner_weights_selected, distanceMatrix, neighborMatrix, epsilon_1/2, epsilon_2/2); 
                
                % Use log convex interporlation function to determine the
                % perturbation matrix for fine-grained locations 
                [z_aipo_instance, loss_aipo_instance(grid_size_index, test_idx)] = logconv_interp(z_anchor_instance, corner_weights_selected, loss_matrix_selected);

                % If the utiltiy loss of the current epsilon_1 and epsilon_2 achieves a new lower utility loss
                % Update the privacy budget allocation to the current
                % (epsilon_1, epsilon_2)
                if loss_aipo_instance(grid_size_index, test_idx) < loss_aipo(grid_size_index, test_idx)                          
                    loss_aipo(grid_size_index, test_idx) = loss_aipo_instance(grid_size_index, test_idx); 
                    z_opt_aipo = z_aipo_instance; 
                end
            end
            time_aipo(grid_size_index, test_idx) = toc;

            % Calculate the mDP violation ratio
            [violation_aipo(grid_size_index, test_idx), ppr_aipo] = compute_mDP_violation(z_opt_aipo(samples_viocheck,:), euclidean_distance_matrix_viocheck, EPSILON);

        end
    end

    save(sprintf("./results/granularity/time/%s/time_aipo.mat", city), "time_aipo"); 
    save(sprintf("./results/granularity/cost/%s/loss_aipo.mat", city), "loss_aipo"); 
    save(sprintf("./results/granularity/time/%s/nr_anchor.mat", city), "nr_anchor"); 

end
