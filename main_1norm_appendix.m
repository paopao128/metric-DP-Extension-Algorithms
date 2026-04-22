%% Header
addpath('./functions/haversine');  
addpath('./functions');
addpath('./functions/benchmarks');


fprintf('------------------- Environment settings --------------------- \n\n');

parameters; 
city_list = {'rome', 'nyc', 'london'};

l_p = 1; % l_1 norm

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


    NR_REAL_LOC = size(col_longitude, 1); 
    
    % Read the utiltiy loss matrix here
    path_file = sprintf('./intermediate/%s/loss_matrix_orig.mat', city);

    load(path_file);

    % Initialize metrics
    time_em = 0; time_laplace = 0; time_tem = 0;
    time_copt = 0; time_lp = 0; time_aipo = 0; time_bound = 0;
    loss_em = 0; loss_laplace = 0; loss_tem = 0;
    loss_copt = 0; loss_lp = 0; loss_aipo = 0; loss_bound = 0;
    violation_em = 0; violation_laplace = 0; violation_tem = 0;
    violation_copt = 0; violation_lp = 0; violation_aipo = 0; violation_bound = 0;
    
    prior = ones(1, NR_REAL_LOC)/NR_REAL_LOC;                               % We consider a case where vehicles are evenly distributed. 


%% ------------------------ Start running the simulation here ------------------------------
    for test_idx = 1:1:NR_TEST                                              % This for loop repeats the experiments for NR_TEST times
        samples_viocheck = randperm(NR_REAL_LOC, SAMPLE_SIZE_PPR);
        perturbed_indices = randi([1, NR_REAL_LOC], 1, NR_PER_LOC);
        loss_matrix = loss_matrix_orig(selected_indices, :); 
        loss_matrix_max = min(loss_matrix, [], 2); 
        [adjMatrix, distanceMatrix, neighborMatrix, cornerPoints, squares, lambda_x, lambda_y, corner_weights] = uniform_anchor(col_latitude, col_longitude, loss_matrix_max, cell_size(1, city_idx));        
        close all; 
        euclidean_distance_matrix_viocheck = distance_matrix(col_latitude(samples_viocheck), col_longitude(samples_viocheck), l_p); 
        for epsilon_idx = 1:1:EPSILON_MAX                                   % This for loop indicates the different epsilon values
            EPSILON = 0.2*epsilon_idx;                                      % Epsilon is changed from 0.2 to 1.6 
            N = NR_REAL_LOC; 
            selected_indices = 1:NR_REAL_LOC;                               % Select all the real locations within the map

            selected_longitudes = col_longitude(selected_indices);
            selected_latitudes = col_latitude(selected_indices);
            selected_osmids = col_osmid(selected_indices);
            loss_matrix_selected = loss_matrix(selected_indices, :); 
            corner_weights_selected = corner_weights(selected_indices, :); 
            

            perturbed_longitudes = col_longitude(perturbed_indices);
            perturbed_latitudes = col_latitude(perturbed_indices);



            %% Lower bound ---------------------------------------------------
            tic 
            % First calculate the utiltiy loss matrix and distance matrix
            % between grid cells
            [grid_utility_loss, grid_distances, grid_prior, ~, ~] = partition_grid_bound(selected_longitudes, selected_latitudes, loss_matrix_selected, prior, GRID_SIZE_LP_LB, GRID_SIZE_LP_LB, col_longitude(perturbed_indices), col_latitude(perturbed_indices), l_p);
            % Then calculate the perturbation matrix using "perturbation_cal_lp"
            [z_bound, loss_bound(epsilon_idx, test_idx)] = perturbation_cal_lp(grid_utility_loss, grid_distances, grid_prior, EPSILON); 


            %% LP (Compared method) ---------------------------------------------------
            tic 

            % First calculate the utiltiy loss matrix and distance matrix
            % between grid cells
            if epsilon_idx >= 3 
                GRID_SIZE_LP_INST = GRID_SIZE_LP;  % This is an adjustment of LP since when epsilon is small, its time complexity is too high. 
            else 
                GRID_SIZE_LP_INST = 12; 
            end
            [grid_utility_loss, grid_distances, grid_prior, ~, ~] = partition_grid(selected_longitudes, selected_latitudes, loss_matrix_selected, prior, GRID_SIZE_LP_INST, GRID_SIZE_LP_INST, col_longitude(perturbed_indices), col_latitude(perturbed_indices), l_p); 

            % Then calculate the perturbation matrix using "perturbation_cal_lp"
            [z_lp, loss_lp(epsilon_idx, test_idx)] = perturbation_cal_lp(grid_utility_loss, grid_distances, grid_prior, EPSILON); 

            % Convert the perturbation matrix for grid celss to the matrix
            % for fine grained locations
            z_lp_fine = convert_grid_to_fine_perturbation(z_lp, col_longitude, col_latitude, GRID_SIZE_LP_INST, GRID_SIZE_LP_INST); 
            time_lp(epsilon_idx, test_idx) = toc; 


            % Calculate the mDP violation ratio
            [violation_lp(epsilon_idx, test_idx), ppr_lp] = compute_mDP_violation(z_lp_fine(samples_viocheck,:), euclidean_distance_matrix_viocheck, EPSILON); 

            %% COPT (Compared method) ---------------------------------------------------
            tic
            
            % First calculate the utiltiy loss matrix and distance matrix
            % between grid cells
            [grid_utility_loss, grid_distances, grid_prior, neighbor_pairs, distances_to_perturbed] = partition_grid(selected_longitudes, selected_latitudes, loss_matrix_selected, prior, GRID_SIZE_COPT, GRID_SIZE_COPT, col_longitude(perturbed_indices), col_latitude(perturbed_indices), l_p); 
            
            % Then calculate the perturbation matrix using "perturbation_cal_lp"
            [z_copt, M, loss_copt(epsilon_idx, test_idx)] = perturbation_cal_copt(distances_to_perturbed, grid_distances, grid_utility_loss, grid_prior, EPSILON, LAMBDA, R);
            time_copt(epsilon_idx, test_idx) = toc; 

            % Convert the perturbation matrix for grid celss to the matrix
            % for fine grained locations
            z_copt_fine = convert_grid_to_fine_perturbation(z_copt, col_longitude, col_latitude, GRID_SIZE_COPT, GRID_SIZE_COPT); 

            % Calculate the mDP violation ratio
            [violation_copt(epsilon_idx, test_idx), ppr_copt] = compute_mDP_violation(z_copt_fine(samples_viocheck,:), euclidean_distance_matrix_viocheck, EPSILON); 
        
            %% EM (Compared method) ---------------------------
            tic

            % Calculate the perturbation matrix using "perturbation_cal_em"
            [z_em, loss_em(epsilon_idx, test_idx)] = perturbation_cal_em(selected_longitudes, selected_latitudes, perturbed_longitudes, perturbed_latitudes, loss_matrix_selected, EPSILON); 
            time_em(epsilon_idx, test_idx) = toc;

            % Calculate the mDP violation ratio
            [violation_em(epsilon_idx, test_idx), ppr_em] = compute_mDP_violation(z_em(samples_viocheck,:), euclidean_distance_matrix_viocheck, EPSILON);

            %% Laplace (Compared method) ---------------------------
            tic

            % Calculate the perturbation matrix using "perturbation_cal_laplace"
            [z_laplace, loss_laplace(epsilon_idx, test_idx)] = perturbation_cal_laplace(selected_longitudes, selected_latitudes, perturbed_longitudes, perturbed_latitudes, loss_matrix_selected, EPSILON); 
            time_laplace(epsilon_idx, test_idx) = toc;

            % Calculate the mDP violation ratio
            [violation_laplace(epsilon_idx, test_idx), ppr_laplace] = compute_mDP_violation(z_laplace(samples_viocheck,:), euclidean_distance_matrix_viocheck, EPSILON);
    
            %% TEM (Compared method) ---------------------------
            tic

            % Calculate the perturbation matrix using "perturbation_cal_tem"
            [z_tem, loss_tem(epsilon_idx, test_idx)] = perturbation_cal_tem(selected_longitudes, selected_latitudes, perturbed_longitudes, perturbed_latitudes, loss_matrix_selected, EPSILON);
            time_tem(epsilon_idx, test_idx) = toc;

            % Calculate the mDP violation ratio
            [violation_tem(epsilon_idx, test_idx), ppr_tem] = compute_mDP_violation(z_tem(samples_viocheck,:), euclidean_distance_matrix_viocheck, EPSILON);

            %% RMP (Compared method) ---------------------------
            tic
            loss_rmp(epsilon_idx, test_idx) = perturbation_cal_rmp(z_em, prior, loss_matrix_selected); 
            time_rmp(epsilon_idx, test_idx) = toc;
            violation_rmp(epsilon_idx, test_idx) = violation_em(epsilon_idx, test_idx); 
            ppr_rmp = ppr_em;

             %% Interporlation
            c_approx = corner_weights_selected'*loss_matrix_selected; 
            tic
            loss_aipo(epsilon_idx, test_idx) = 999999999;
            for epsilon_idx_1 = 1:1:NR_EPSILON_INTERVAL-1                   % Discretize epsilon_1 and try it one by one
                epsilon_1 = EPSILON*epsilon_idx_1/NR_EPSILON_INTERVAL; 
                epsilon_2 = EPSILON*sqrt(1-(epsilon_idx_1/NR_EPSILON_INTERVAL)^2);           % Given epsilon 1, calculate the corresponding epsilon_2

                % Calculate the perturbation matrix for anchor records using "perturbation_cal_apo"
                z_anchor_instance = perturbation_cal_apo(c_approx, corner_weights_selected, distanceMatrix, neighborMatrix, epsilon_1/2, epsilon_2/2); 
                
                % Use log convex interporlation function to determine the
                % perturbation matrix for fine-grained locations 
                [z_aipo_instance, loss_aipo_instance(epsilon_idx, test_idx, epsilon_idx_1)] = logconv_interp(z_anchor_instance, corner_weights_selected, loss_matrix_selected);

                % If the utiltiy loss of the current epsilon_1 and epsilon_2 achieves a new lower utility loss
                % Update the privacy budget allocation to the current
                % (epsilon_1, epsilon_2)
                if loss_aipo_instance(epsilon_idx, test_idx, epsilon_idx_1) < loss_aipo(epsilon_idx, test_idx)                          
                    loss_aipo(epsilon_idx, test_idx) = loss_aipo_instance(epsilon_idx, test_idx, epsilon_idx_1); 
                    z_opt_aipo = z_aipo_instance; 
                end
            end
            time_aipo(epsilon_idx, test_idx) = toc;

            % Calculate the mDP violation ratio
            [violation_aipo(epsilon_idx, test_idx), ppr_aipo] = compute_mDP_violation(z_opt_aipo(samples_viocheck,:), euclidean_distance_matrix_viocheck, EPSILON);
            fprintf('Current progress: Test index is %d; epsilon value = %f km^-1 ...\n', test_idx, epsilon_idx*0.2);

        end
    end

    %% Save results with city-specific paths
    save(sprintf("./results/1norm/time/%s/time_em.mat", city), "time_em"); 
    save(sprintf("./results/1norm/time/%s/time_laplace.mat", city), "time_laplace"); 
    save(sprintf("./results/1norm/time/%s/time_tem.mat", city), "time_tem"); 
    save(sprintf("./results/1norm/time/%s/time_copt.mat", city), "time_copt"); 
    save(sprintf("./results/1norm/time/%s/time_lp.mat", city), "time_lp"); 
    save(sprintf("./results/1norm/time/%s/time_aipo.mat", city), "time_aipo"); 
    save(sprintf("./results/1norm/time/%s/time_rmp.mat", city), "time_rmp"); 

    save(sprintf("./results/1norm/cost/%s/loss_em.mat", city), "loss_em"); 
    save(sprintf("./results/1norm/cost/%s/loss_laplace.mat", city), "loss_laplace"); 
    save(sprintf("./results/1norm/cost/%s/loss_tem.mat", city), "loss_tem"); 
    save(sprintf("./results/1norm/cost/%s/loss_copt.mat", city), "loss_copt"); 
    save(sprintf("./results/1norm/cost/%s/loss_lp.mat", city), "loss_lp"); 
    save(sprintf("./results/1norm/cost/%s/loss_aipo.mat", city), "loss_aipo"); 
    save(sprintf("./results/1norm/cost/%s/loss_bound.mat", city), "loss_bound"); 
    save(sprintf("./results/1norm/cost/%s/loss_rmp.mat", city), "loss_rmp"); 

    save(sprintf("./results/1norm/violation/%s/violation_em.mat", city), "violation_em"); 
    save(sprintf("./results/1norm/violation/%s/violation_laplace.mat", city), "violation_laplace"); 
    save(sprintf("./results/1norm/violation/%s/violation_tem.mat", city), "violation_tem"); 
    save(sprintf("./results/1norm/violation/%s/violation_copt.mat", city), "violation_copt"); 
    save(sprintf("./results/1norm/violation/%s/violation_lp.mat", city), "violation_lp"); 
    save(sprintf("./results/1norm/violation/%s/violation_aipo.mat", city), "violation_aipo"); 
    save(sprintf("./results/1norm/violation/%s/violation_rmp.mat", city), "violation_rmp"); 

    save(sprintf("./results/1norm/ppr/%s/ppr_em.mat", city), "ppr_em"); 
    save(sprintf("./results/1norm/ppr/%s/ppr_laplace.mat", city), "ppr_laplace"); 
    save(sprintf("./results/1norm/ppr/%s/ppr_tem.mat", city), "ppr_tem"); 
    save(sprintf("./results/1norm/ppr/%s/ppr_copt.mat", city), "ppr_copt"); 
    save(sprintf("./results/1norm/ppr/%s/ppr_lp.mat", city), "ppr_lp"); 
    save(sprintf("./results/1norm/ppr/%s/ppr_aipo.mat", city), "ppr_aipo"); 
    save(sprintf("./results/1norm/ppr/%s/ppr_rmp.mat", city), "ppr_rmp");

end
