%% Header
addpath('./functions/haversine');  
addpath('./functions');
addpath('./functions/benchmarks');


fprintf('------------------- Environment settings --------------------- \n\n');

parameters; 
city_list = {'rome', 'nyc', 'london'};

l_p = 2; % l_2 norm

for city_idx = 1:1 % length(city_list)
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
    time_em = 0; time_laplace = 0; time_tem = 0; time_rmp =0;
    time_copt = 0; time_lp = 0; time_aipo = 0; time_aipor = 0; time_bound = 0;
    loss_em = 0; loss_laplace = 0; loss_tem = 0; loss_rmp =0;
    loss_copt = 0; loss_lp = 0; loss_aipo = 0; loss_aipor = 0; loss_bound = 0;
    violation_em = 0; violation_laplace = 0; violation_tem = 0; violation_rmp = 0; 
    violation_copt = 0; violation_lp = 0; violation_aipo = 0; violation_aipor = 0; violation_bound = 0;
    
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
            EPSILON = 0.5*epsilon_idx;                                      % Epsilon is changed from 0.2 to 1.6 
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
            % [grid_utility_loss, grid_distances, grid_prior, ~, ~] = partition_grid_bound(selected_longitudes, selected_latitudes, loss_matrix_selected, prior, GRID_SIZE_LP_LB, GRID_SIZE_LP_LB, col_longitude(perturbed_indices), col_latitude(perturbed_indices), l_p);
            % % Then calculate the perturbation matrix using "perturbation_cal_lp"
            % [z_bound, loss_bound(epsilon_idx, test_idx)] = perturbation_cal_lp(grid_utility_loss, grid_distances, grid_prior, EPSILON); 


           
             %% Interporlation
            c_approx = corner_weights_selected'*loss_matrix_selected; 
            tic
            loss_aipo(epsilon_idx, test_idx) = 999999999;
            for epsilon_idx_1 = 1:1:NR_EPSILON_INTERVAL-1                   % Discretize epsilon_1 and try it one by one
                epsilon_1 = EPSILON*epsilon_idx_1/NR_EPSILON_INTERVAL; 
                epsilon_2 = EPSILON*sqrt(1-(epsilon_idx_1/NR_EPSILON_INTERVAL)^2);           % Given epsilon 1, calculate the corresponding epsilon_2

                % Calculate the perturbation matrix for anchor records using "perturbation_cal_apo"
                [z_anchor_instance, loss_aipo_instance_approx(epsilon_idx, test_idx, epsilon_idx_1)] = perturbation_cal_apo(c_approx, corner_weights_selected, distanceMatrix, neighborMatrix, epsilon_1/2, epsilon_2/2); 
                loss_aipo_instance_approx(epsilon_idx, test_idx, epsilon_idx_1) = loss_aipo_instance_approx(epsilon_idx, test_idx, epsilon_idx_1)/NR_REAL_LOC;
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
            % 
            % 
           

            fprintf('Current progress: Test index is %d; epsilon value = %f km^-1 ...\n', test_idx, EPSILON);

        end
    end

    %% Save results with city-specific paths
    save(sprintf("./results/2norm/time/%s/time_em.mat", city), "time_em"); 
    save(sprintf("./results/2norm/time/%s/time_laplace.mat", city), "time_laplace"); 
    save(sprintf("./results/2norm/time/%s/time_tem.mat", city), "time_tem"); 
    save(sprintf("./results/2norm/time/%s/time_copt.mat", city), "time_copt"); 
    save(sprintf("./results/2norm/time/%s/time_lp.mat", city), "time_lp"); 
    save(sprintf("./results/2norm/time/%s/time_aipo.mat", city), "time_aipo"); 
    save(sprintf("./results/2norm/time/%s/time_aipor.mat", city), "time_aipor"); 
    save(sprintf("./results/2norm/time/%s/time_rmp.mat", city), "time_rmp"); 

    save(sprintf("./results/2norm/cost/%s/loss_em.mat", city), "loss_em"); 
    save(sprintf("./results/2norm/cost/%s/loss_laplace.mat", city), "loss_laplace"); 
    save(sprintf("./results/2norm/cost/%s/loss_tem.mat", city), "loss_tem"); 
    save(sprintf("./results/2norm/cost/%s/loss_copt.mat", city), "loss_copt"); 
    save(sprintf("./results/2norm/cost/%s/loss_lp.mat", city), "loss_lp"); 
    save(sprintf("./results/2norm/cost/%s/loss_aipo.mat", city), "loss_aipo"); 
    save(sprintf("./results/2norm/cost/%s/loss_aipor.mat", city), "loss_aipor"); 
    save(sprintf("./results/2norm/cost/%s/loss_bound.mat", city), "loss_bound"); 
    save(sprintf("./results/2norm/cost/%s/loss_rmp.mat", city), "loss_rmp"); 

    save(sprintf("./results/2norm/violation/%s/violation_em.mat", city), "violation_em"); 
    save(sprintf("./results/2norm/violation/%s/violation_laplace.mat", city), "violation_laplace"); 
    save(sprintf("./results/2norm/violation/%s/violation_tem.mat", city), "violation_tem"); 
    save(sprintf("./results/2norm/violation/%s/violation_copt.mat", city), "violation_copt"); 
    save(sprintf("./results/2norm/violation/%s/violation_lp.mat", city), "violation_lp"); 
    save(sprintf("./results/2norm/violation/%s/violation_aipo.mat", city), "violation_aipo"); 
    save(sprintf("./results/2norm/violation/%s/violation_aipor.mat", city), "violation_aipor"); 
    save(sprintf("./results/2norm/violation/%s/violation_rmp.mat", city), "violation_rmp"); 

    % save(sprintf("./results/1norm/ppr/%s/ppr_em.mat", city), "ppr_em"); 
    % save(sprintf("./results/1norm/ppr/%s/ppr_laplace.mat", city), "ppr_laplace"); 
    % save(sprintf("./results/1norm/ppr/%s/ppr_tem.mat", city), "ppr_tem"); 
    % save(sprintf("./results/1norm/ppr/%s/ppr_copt.mat", city), "ppr_copt"); 
    % save(sprintf("./results/1norm/ppr/%s/ppr_lp.mat", city), "ppr_lp"); 
    % save(sprintf("./results/1norm/ppr/%s/ppr_aipo.mat", city), "ppr_aipo"); 
    % save(sprintf("./results/1norm/ppr/%s/ppr_rmp.mat", city), "ppr_rmp");

    %% Print the results 
    
    print_loss_table(city,   ...
    loss_em/1000, loss_laplace/1000, loss_tem/1000, ...
    loss_rmp/1000, loss_copt/1000, loss_lp/1000, ...
    loss_aipor/1000, loss_bound/1000, loss_aipo/1000);

    print_violation_table(city, ...
    violation_em*100, violation_laplace*100, violation_tem*100, ...
    violation_rmp*100, violation_copt*100, violation_lp*100, ...
    violation_aipor*100, violation_aipo*100);

    print_time_table(city, ...
    time_copt, time_lp, time_aipo);
end

