%% main_PANDA_exp.m
% Run PAnDA algorithm using the same three-layer point sets as the tree methods.
% Layer k uses the allPoints produced by k rounds of interpolation_extension_k
% (depth-2: ~483 pts, depth-3: ~1845 pts, depth-4: ~7209 pts).
% The same dataset, SCALE filtering, and lonlat_to_xy conversion are applied
% as in all other tree-based methods.
%
% Output: 9 x.xx±x.xx values for loss and computation time
%         (EPSILON_MAX=3 epsilon values x 3 layers, column-major).

addpath('./functions/haversine');
addpath('./functions');
addpath('./PAnDA code/functions/');
addpath('./PAnDA code/functions/myBDToolbox');
addpath('./PAnDA code/functions/haversine');
addpath('./PAnDA code/functions/myPlotToolbox');

fprintf('------------------- Environment settings --------------------- \n\n');

parameters;                        % NR_TEST, EPSILON_MAX, NR_PER_LOC, SCALE, cell_size, ...
run('./PAnDA code/parameters.m');  % env_parameters struct (NR_AGENT=15, NR_OBFLOC=100, ...)

city_list = {'rome', 'nyc', 'london'};

% Fixed PAnDA hyper-parameters
lambda    = 0.04;
alpha_hat = 0.95;
delta     = 1e-9;
ITER      = env_parameters.ITER;
NR_AGENT  = env_parameters.NR_AGENT;    % 15  (k-means clusters for Benders)
NR_OBFLOC = env_parameters.NR_OBFLOC;  % 100 (candidate output locations)

for city_idx = 1:1
    city = city_list{city_idx};
    fprintf('\n------------- Processing city: %s -------------\n', city);

    %% Load tree_artifact dataset (same as all tree methods)
    node_file = sprintf('./datasets/%s/nodes.csv', city);
    edge_file  = sprintf('./datasets/%s/edges.csv', city);

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
    mid_latitude  = (max_latitude + min_latitude) / 2;
    LATITUDE_SIZE = max_latitude - min_latitude;

    lon_range = [mid_longitude - LONGITUDE_SIZE/SCALE, mid_longitude + LONGITUDE_SIZE/SCALE];
    lat_range = [mid_latitude  - LATITUDE_SIZE/SCALE,  mid_latitude  + LATITUDE_SIZE/SCALE];

    selected_indices = filter_coords_by_range(col_longitude_orig, col_latitude_orig, lon_range, lat_range);
    col_longitude = col_longitude_orig(selected_indices, :);
    col_latitude  = col_latitude_orig(selected_indices, :);
    [col_longitude, col_latitude] = lonlat_to_xy(col_longitude, col_latitude, mid_longitude, mid_latitude);

    NR_REAL_LOC = size(col_longitude, 1);
    fprintf('Map loaded: %d nodes (after filtering).\n', NR_REAL_LOC);

    %% Accumulators: (EPSILON_MAX x 3_layers x NR_TEST)
    loss_all = zeros(EPSILON_MAX, 3, NR_TEST);
    time_all = zeros(NR_TEST, EPSILON_MAX, 3);

    for test_idx = 1:NR_TEST
        fprintf('  Test %d/%d ...\n', test_idx, NR_TEST);

        %% Same random selection as tree methods
        perturbed_indices = randi([1, NR_REAL_LOC], 1, NR_PER_LOC);
        perturbed_xy      = [col_latitude(perturbed_indices', 1), col_longitude(perturbed_indices', 1)];
        loss_matrix       = pdist2([col_latitude, col_longitude], perturbed_xy);
        loss_matrix_max   = min(loss_matrix, [], 2);

        %% Build anchor structure (same as tree methods)
        [~, ~, ~, cornerPoints_ori, ~, ~, ~, ~] = ...
            uniform_anchor(col_latitude, col_longitude, loss_matrix_max, cell_size(1, city_idx));
        close all;

        %% Produce allPoints for each of the 3 interpolation layers
        allPoints_by_layer = cell(1, 3);
        tmp_allPoints = cornerPoints_ori;
        for interp_id = 1:3
            cornerPoints = tmp_allPoints;
            interpolation_extension_k;          % produces allPoints, W
            allPoints_by_layer{interp_id} = allPoints;
            tmp_allPoints = allPoints;
        end

        %% Run PAnDA on each layer
        for layer_id = 1:1
            pts  = allPoints_by_layer{layer_id};  % N_pts x 2, km [lat, lon]
            N_pts = size(pts, 1);
            NR_OBFLOC_EFF = min(NR_OBFLOC, N_pts);
            NR_AGENT_EFF  = min(NR_AGENT,  N_pts);

            %% Distance matrix in km (pts already in km -- use Euclidean pdist2)
            dist_mat = pdist2(pts, pts);   % N_pts x N_pts
            adj_mat  = heaviside(1 - dist_mat / env_parameters.NEIGHBOR_THRESHOLD);

            D_MAX_l          = max(max(dist_mat));
            range_thr_l      = D_MAX_l / 10;
            threshold_mat_l  = dist_mat < range_thr_l;

            %% Random agents and output locations (drawn from allPoints)
            user_l    = randperm(N_pts, NR_AGENT_EFF);   % 1 x NR_AGENT_EFF
            obf_loc_l = randperm(N_pts, NR_OBFLOC_EFF);  % 1 x NR_OBFLOC_EFF

            %% Mobility model
            w_l = getq(dist_mat, lambda, range_thr_l, alpha_hat);
            [~, all_target_l]    = get_relevant_location_set(w_l, user_l);
            epsilon_nmw_l        = get_epsilon_nmw(w_l, all_target_l, dist_mat);
            B_xn_xnhat_l         = get_B(w_l, dist_mat);

            %% Pr matrix (expensive, once per test x layer)
            Pr_l = zeros(N_pts, N_pts);
            for n_hat_ind = 1:length(all_target_l)
                n_hat = all_target_l(n_hat_ind);
                for m_hat_ind = 1:length(all_target_l)
                    m_hat = all_target_l(m_hat_ind);
                    if n_hat ~= m_hat
                        num_nn = length(find(dist_mat(n_hat, :) < range_thr_l));
                        [~, sIdx] = sort(dist_mat(n_hat, :), 'ascend');
                        c_nhat = sIdx(1:num_nn);

                        num_nm = length(find(dist_mat(m_hat, :) < range_thr_l));
                        [~, sIdx] = sort(dist_mat(m_hat, :), 'ascend');
                        c_mhat = sIdx(1:num_nm);

                        for ni = 1:num_nn
                            for mi = 1:num_nm
                                s_n = sum(w_l(:, n_hat));
                                s_m = sum(w_l(:, m_hat));
                                Pr_nm = w_l(c_nhat(ni), n_hat) * w_l(c_mhat(mi), m_hat) / s_n / s_m;
                                Pr_l(n_hat, m_hat) = Pr_l(n_hat, m_hat) + Pr_nm;
                            end
                        end
                    end
                end
            end

            %% Filter to all_target subset
            [adj_filt, dist_filt, epsilon_nmw_filt] = reget( ...
                adj_mat, dist_mat, all_target_l, epsilon_nmw_l);
            NR_NODE_FILT = length(all_target_l);

            %% Cost matrix: direct km distance between input and output (consistent with tree methods)
            cost_mat_l = dist_mat(all_target_l, obf_loc_l) / NR_NODE_FILT;

            env_params                    = env_parameters;
            env_params.NR_NODE_IN_TARGET  = NR_NODE_FILT;
            env_params.NR_OBFLOC          = NR_OBFLOC_EFF;
            env_params.cost_matrix        = cost_mat_l;

            % node_in_target passed to agentCreation only for its size
            node_in_target_filt = 1:NR_NODE_FILT;
            cluster_idx = kmeans(dist_filt, NR_AGENT_EFF);

            %% Per-epsilon loop
            for epsilon_idx = 1:EPSILON_MAX
                EPSILON = 0.5 * epsilon_idx;
                env_params.EPSILON = EPSILON;

                t_start = tic;

                [xi_hathat, ~] = get_xi_hathat(dist_mat, epsilon_nmw_l, EPSILON, ...
                    user_l, delta, w_l, Pr_l, all_target_l, threshold_mat_l, ...
                    B_xn_xnhat_l, range_thr_l);

                agent_2PPO = agentCreation(cluster_idx, node_in_target_filt, ...
                    adj_filt, dist_filt, NR_AGENT_EFF, NR_NODE_FILT, NR_OBFLOC_EFF, ...
                    EPSILON, epsilon_nmw_filt, xi_hathat);

                masteragent = masterAgentCreation(dist_filt, agent_2PPO, ...
                    adj_filt, cluster_idx, NR_NODE_FILT, NR_OBFLOC_EFF, NR_AGENT_EFF, ...
                    EPSILON, epsilon_nmw_filt, xi_hathat);

                [~, ~, ~, ~, ~, ~, obf_matrix] = bendersDecomposition( ...
                    masteragent, agent_2PPO, env_params, ITER);

                time_all(test_idx, epsilon_idx, layer_id) = toc(t_start);

                loss_mat_val = cost_mat_l .* obf_matrix;
                loss_all(epsilon_idx, layer_id, test_idx) = sum(loss_mat_val(:));
            end
        end  % layer_id

        fprintf('    Test %d done.\n', test_idx);
    end  % test_idx

    %% Output: 9 values (EPSILON_MAX x 3 layers), column-major [L1E1,L1E2,L1E3,L2E1,...]

    % --- Loss ---
    mean_loss = squeeze(mean(loss_all, 3));   % EPSILON_MAX x 3
    std_loss  = squeeze(std(loss_all,  0, 3));
    all_mean_l = mean_loss(:);   % 9 x 1
    all_std_l  = std_loss(:);
    parts_l = cell(1, 9);
    for k = 1:9
        parts_l{k} = sprintf('%.2f%s%.2f', all_mean_l(k), char(177), all_std_l(k));
    end
    fprintf('[%s] loss: %s\n', city, strjoin(parts_l, ' & '));

    % --- Computation time ---
    mean_time = squeeze(mean(time_all, 1));   % EPSILON_MAX x 3
    std_time  = squeeze(std(time_all,  0, 1));
    all_mean_t = mean_time(:);   % 9 x 1
    all_std_t  = std_time(:);
    parts_t = cell(1, 9);
    for k = 1:9
        parts_t{k} = sprintf('%.2f%s%.2f', all_mean_t(k), char(177), all_std_t(k));
    end
    fprintf('[%s] time: %s\n', city, strjoin(parts_t, ' & '));

    % --- Append to results file ---
    fid = fopen('results_PANDA.txt', 'a');
    fprintf(fid, 'loss [%s]: %s\n', city, strjoin(parts_l, ' & '));
    fprintf(fid, 'time [%s]: %s\n', city, strjoin(parts_t, ' & '));
    fclose(fid);

end  % city_idx
