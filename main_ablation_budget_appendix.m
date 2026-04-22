%% Header
addpath('./functions/haversine');  
addpath('./functions');
addpath('./functions/benchmarks');
l_p = 2; 

fprintf('------------------- Environment settings --------------------- \n\n');

parameters; 
city_list = {'rome', 'nyc', 'london'};

l_p = 2; % l_2 norm

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
    time_exp = 0; time_laplace = 0; time_tem = 0;
    time_copt = 0; time_lp = 0; time_aipo = 0; time_aipor = 0; time_bound = 0;
    loss_exp = 0; loss_laplace = 0; loss_tem = 0;
    loss_copt = 0; loss_lp = 0; loss_aipo = 0; loss_aipor = 0; loss_bound = 0;
    violation_exp = 0; violation_laplace = 0; violation_tem = 0;
    violation_copt = 0; violation_lp = 0; violation_aipo = 0; violation_aipor = 0; violation_bound = 0;
    
    prior = ones(1, NR_REAL_LOC)/NR_REAL_LOC;                               % We consider a case where vehicles are evenly distributed. 


%% ------------------------ Start running the simulation here ------------------------------
    for test_idx = 1:1:NR_TEST                                              % This for loop repeats the experiments for NR_TEST times
        samples_viocheck = randperm(NR_REAL_LOC, SAMPLE_SIZE_PPR);
        perturbed_indices = randi([1, NR_REAL_LOC], 1, NR_PER_LOC);
        loss_matrix = loss_matrix_orig(selected_indices, :); 
        loss_matrix_max = min(loss_matrix, [], 2); 
        [adjMatrix, distanceMatrix, neighborMatrix, cornerPoints, squares, lambda_x, lambda_y, corner_weights] = uniform_anchor(col_latitude, col_longitude, loss_matrix_max, cell_size(1, city_idx));
        hold on;
        scatter(col_latitude(perturbed_indices), col_longitude(perturbed_indices)); 
        
        
        close all; 
        euclidean_distance_matrix_viocheck = distance_matrix(col_latitude(samples_viocheck), col_longitude(samples_viocheck), l_p); 
        for epsilon_idx = 1:1:8                                             % This for loop indicates the different epsilon values
            EPSILON = 0.2*epsilon_idx;            
            N = NR_REAL_LOC; 
            selected_indices = 1:NR_REAL_LOC; 

            selected_longitudes = col_longitude(selected_indices);
            selected_latitudes = col_latitude(selected_indices);
            selected_osmids = col_osmid(selected_indices);
            loss_matrix_selected = loss_matrix(selected_indices, :); 
            corner_weights_selected = corner_weights(selected_indices, :); 

            perturbed_longitudes = col_longitude(perturbed_indices);
            perturbed_latitudes = col_latitude(perturbed_indices);
            
            c_approx = corner_weights_selected'*loss_matrix_selected; 

            %% INT without privacy budget optimization ---------------------------
            z_anchor_aipoe = perturbation_cal_apo(c_approx, corner_weights_selected, distanceMatrix, neighborMatrix, EPSILON/sqrt(8), EPSILON/sqrt(8)); 
            [z_aipoe, loss_aipoe(epsilon_idx, test_idx)] = logconv_interp(z_anchor_aipoe, corner_weights_selected, loss_matrix_selected);
            [violation_aipoe(epsilon_idx, test_idx), ppr_aipoe] = compute_mDP_violation(z_aipoe(samples_viocheck,:), euclidean_distance_matrix_viocheck, EPSILON);


            %% INT (Our method) ---------------------------
            
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
                         
        end
    end

    %% Save results with city-specific paths
    save(sprintf("./results/ablation_privacybudget/cost/%s/loss_aipo_instance.mat", city), "loss_aipo_instance"); 
    save(sprintf("./results/ablation_privacybudget/cost/%s/loss_aipo.mat", city), "loss_aipo"); 
    save(sprintf("./results/ablation_privacybudget/cost/%s/loss_aipoe.mat", city), "loss_aipoe"); 
end
