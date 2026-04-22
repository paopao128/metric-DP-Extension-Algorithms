%% Header
addpath('./functions/haversine');  
addpath('./functions');
addpath('./functions/benchmarks');


fprintf('------------------- Environment settings --------------------- \n\n');

parameters; 
city_list = {'rome', 'nyc', 'london'};

l_p = 2; % l_2 norm

for city_idx = 3:3 % length(city_list)
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
    loss_copt = 0; loss_lp = 0; loss_MWE = 0; loss_aipor = 0; loss_bound = 0;
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
        [adjMatrix, distanceMatrix, neighborMatrix, cornerPoints_ori, squares, lambda_x, lambda_y, corner_weights] = uniform_anchor(col_latitude, col_longitude, loss_matrix_max, cell_size(1, city_idx));        
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
            corner_weights_selected_ori = corner_weights(selected_indices, :); 
            corner_weights_selected=corner_weights_selected_ori;

            perturbed_longitudes = col_longitude(perturbed_indices);
            perturbed_latitudes = col_latitude(perturbed_indices);

            % Prior for each original anchor: fraction of real locations it represents.
            anchor_prior = full(corner_weights_selected_ori)' * ones(NR_REAL_LOC, 1) / NR_REAL_LOC;


            %% Interporlation

            c_approx = corner_weights_selected'*loss_matrix_selected;
           
            
            loss_MWE(epsilon_idx, test_idx) = 999999999;
            t_apo_plus_1 = 0; t_incr_2 = 0; t_incr_3 = 0;
            for epsilon_idx_1 = 1:1:NR_EPSILON_INTERVAL-1                   % Discretize epsilon_1 and try it one by one
                epsilon_1 = EPSILON*epsilon_idx_1/NR_EPSILON_INTERVAL;
                epsilon_2 = EPSILON*sqrt(1-(epsilon_idx_1/NR_EPSILON_INTERVAL)^2);           % Given epsilon 1, calculate the corresponding epsilon_2
                corner_weights_selected=corner_weights_selected_ori;
                cornerPoints=cornerPoints_ori;
                % Calculate the perturbation matrix for anchor records using "perturbation_cal_apo"
                t_apo_start = tic;
                [z_anchor_instance, loss_aipo_instance_approx(epsilon_idx, test_idx, epsilon_idx_1)] = perturbation_cal_apo(c_approx, corner_weights_selected, distanceMatrix, neighborMatrix, epsilon_1/2, epsilon_2/2);
                loss_aipo_instance_approx(epsilon_idx, test_idx, epsilon_idx_1) = loss_aipo_instance_approx(epsilon_idx, test_idx, epsilon_idx_1)/NR_REAL_LOC;
                % Use log convex interporlation function to determine the
                % perturbation matrix for fine-grained locations
                [z_aipo_instance_43160, loss_aipo_instance_43160(epsilon_idx, test_idx, epsilon_idx_1)] = logconv_interp(z_anchor_instance, corner_weights_selected, loss_matrix_selected);
                t_apo_this = toc(t_apo_start);
                interp_times=3;
                allPoints=cornerPoints;
                current_prior = anchor_prior;  % start from original anchor prior
                for interp_id=1:interp_times
                    cornerPoints=allPoints;
                    t_interp_start = tic;
                    interpolation_extension_k;
                    corner_weights_selected=W;
                    % Propagate prior to finer grid via interpolation weights W (N_fine x N_coarse)
                    current_prior = full(W) * current_prior;
                    current_prior = current_prior / sum(current_prior);  % renormalize
                    loss_matrix_selected = pdist2(allPoints, perturbed_xy);
                    if interp_id>1
                        z_anchor_instance=z_extension;
                    end
                    % McShane–Whitney extension with density-weighted prior
                    [z_extension,loss_MWE_instance(epsilon_idx, test_idx, epsilon_idx_1)] = MWE_cell_BFS(cornerPoints, z_anchor_instance, allPoints, 0.5, loss_matrix_selected, current_prior);
                    t_interp_step = toc(t_interp_start);
                    if interp_id == 1
                        t_apo_plus_1 = t_apo_plus_1 + t_apo_this + t_interp_step;
                    elseif interp_id == 2
                        t_incr_2 = t_incr_2 + t_interp_step;
                    else
                        t_incr_3 = t_incr_3 + t_interp_step;
                    end
                    loss_MWE_times(epsilon_idx, interp_id, epsilon_idx_1)=loss_MWE_instance(epsilon_idx, test_idx, epsilon_idx_1);

                end
                % If the utiltiy loss of the current epsilon_1 and epsilon_2 achieves a new lower utility loss
                % Update the privacy budget allocation to the current
                % (epsilon_1, epsilon_2)
                if loss_MWE_instance(epsilon_idx, test_idx, epsilon_idx_1) < loss_MWE(epsilon_idx, test_idx)
                    loss_MWE(epsilon_idx, test_idx) = loss_MWE_instance(epsilon_idx, test_idx, epsilon_idx_1);
                    z_opt_MWE = z_extension;
                end
            end
            time_per_eps(epsilon_idx, :) = [t_apo_plus_1, t_apo_plus_1+t_incr_2, t_apo_plus_1+t_incr_2+t_incr_3];
            fprintf('Current progress: Test index is %d; epsilon value = %f km^-1 ...\n', test_idx, EPSILON);

        end
        best_losses_all_tests(test_idx, :, :) = min(loss_MWE_times, [], 3);
        time_all_tests(test_idx, :, :) = time_per_eps;
    end
end
mean_loss = squeeze(mean(best_losses_all_tests, 1));
std_loss  = squeeze(std(best_losses_all_tests, 0, 1));
all_mean = mean_loss(:);
all_std  = std_loss(:);
parts = cell(1, 9);
for k = 1:9
    parts{k} = sprintf('%.2f%s%.2f', all_mean(k), char(177), all_std(k));
end
fprintf('%s\n', strjoin(parts, ' & '));

mean_time = squeeze(mean(time_all_tests, 1));  % EPSILON_MAX x 3
std_time  = squeeze(std(time_all_tests, 0, 1));
all_mean_t = mean_time(:);
all_std_t  = std_time(:);
parts_t = cell(1, 9);
for k = 1:9
    parts_t{k} = sprintf('%.2f%s%.2f', all_mean_t(k), char(177), all_std_t(k));
end
fprintf('tree_MWE [%s] time: %s\n', city, strjoin(parts_t, ' & '));
fid = fopen('results_time.txt', 'a');
fprintf(fid, 'tree_MWE [%s]: %s\n', city, strjoin(parts_t, ' & '));
fclose(fid);
