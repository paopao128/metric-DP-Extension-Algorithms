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
    
    % Initialize metrics
    time_em = 0; time_laplace = 0; time_tem = 0; time_rmp =0;
    time_copt = 0; time_lp = 0; time_aipo = 0; time_aipor = 0; time_bound = 0;
    loss_em = 0; loss_laplace = 0; loss_tem = 0; loss_rmp =0;
    loss_copt = 0; loss_lp = 0; loss_aipo = 0; loss_aipor = 0; loss_bound = 0;
    violation_em = 0; violation_laplace = 0; violation_tem = 0; violation_rmp = 0; 
    violation_copt = 0; violation_lp = 0; violation_aipo = 0; violation_aipor = 0; violation_bound = 0;
    
    prior = ones(1, NR_REAL_LOC)/NR_REAL_LOC;                               % We consider a case where vehicles are evenly distributed. 

    ori_xy=[col_latitude,col_longitude];
%% ------------------------ Start running the simulation here ------------------------------
    for test_idx = 1:1:NR_TEST                                              % This for loop repeats the experiments for NR_TEST times
        samples_viocheck = randperm(NR_REAL_LOC, SAMPLE_SIZE_PPR);
        perturbed_indices = randi([1, NR_REAL_LOC], 1, NR_PER_LOC);
        perturbed_xy = [col_latitude(perturbed_indices', 1), col_longitude(perturbed_indices', 1)];
        loss_matrix = pdist2([col_latitude, col_longitude], perturbed_xy);
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

           
            %% Interporlation
            
            c_approx = corner_weights_selected'*loss_matrix_selected; 
            
            loss_aipo(epsilon_idx, test_idx) = 999999999;
            for epsilon_idx_1 = 1:1:NR_EPSILON_INTERVAL-1                   % Discretize epsilon_1 and try it one by one
                epsilon_1 = EPSILON*epsilon_idx_1/NR_EPSILON_INTERVAL; 
                epsilon_2 = EPSILON*sqrt(1-(epsilon_idx_1/NR_EPSILON_INTERVAL)^2);           % Given epsilon 1, calculate the corresponding epsilon_2

                % Calculate the perturbation matrix for anchor records using "perturbation_cal_apo"
                [z_anchor_instance, loss_aipo_instance_approx(epsilon_idx, test_idx, epsilon_idx_1)] = perturbation_cal_apo(c_approx, corner_weights_selected, distanceMatrix, neighborMatrix, epsilon_1/2, epsilon_2/2); 
                loss_aipo_instance_approx(epsilon_idx, test_idx, epsilon_idx_1) = loss_aipo_instance_approx(epsilon_idx, test_idx, epsilon_idx_1)/NR_REAL_LOC;
                % Use log convex interporlation function to determine the
                % perturbation matrix for fine-grained locations 
                [z_aipo_instance_43160, loss_aipo_instance_43160(epsilon_idx, test_idx, epsilon_idx_1)] = logconv_interp(z_anchor_instance, corner_weights_selected, loss_matrix_selected);
                interp_times=3;
                allPoints=cornerPoints;
                for interp_id=1:interp_times
                    cornerPoints=allPoints;
                    interpolation_extension_k;
                    corner_weights_selected=W;
                    loss_matrix_selected = pdist2(allPoints, perturbed_xy);
                    if interp_id>1
                        z_anchor_instance=z_extension;
                    end
                    % % Log-convex interpolation
                    % [z_extension, loss_aipo_instance(epsilon_idx, test_idx, epsilon_idx_1)] = logconv_interp(z_anchor_instance, corner_weights_selected, loss_matrix_selected);
                    % loss_interp_times(interp_id)=loss_aipo_instance(epsilon_idx, test_idx, epsilon_idx_1);

                    % McShane–Whitney extension
                    [z_extension,loss_MW(epsilon_idx, test_idx, epsilon_idx_1)] = mcshane_whitney_extension(cornerPoints, z_anchor_instance, allPoints, 0.5, loss_matrix_selected);
                    loss_interp_times(interp_id)=loss_MW;
                end
                % If the utiltiy loss of the current epsilon_1 and epsilon_2 achieves a new lower utility loss
                % Update the privacy budget allocation to the current
                % (epsilon_1, epsilon_2)
                if loss_aipo_instance(epsilon_idx, test_idx, epsilon_idx_1) < loss_aipo(epsilon_idx, test_idx)                          
                    loss_aipo(epsilon_idx, test_idx) = loss_aipo_instance(epsilon_idx, test_idx, epsilon_idx_1); 
                    z_opt_aipo = z_extension; 
                end
            end
            fprintf('Current progress: Test index is %d; epsilon value = %f km^-1 ...\n', test_idx, epsilon_idx*0.2);

        end
    end
end

