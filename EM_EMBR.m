%% Header
addpath('./functions/haversine');  
addpath('./functions');
addpath('./functions/benchmarks');
addpath('./all_points');

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
    loss_copt = 0; loss_lp = 0; loss_aipo = 0; loss_aipor = 0; loss_bound = 0;
    violation_em = 0; violation_laplace = 0; violation_tem = 0; violation_rmp = 0; 
    violation_copt = 0; violation_lp = 0; violation_aipo = 0; violation_aipor = 0; violation_bound = 0;
    
    prior = ones(1, NR_REAL_LOC)/NR_REAL_LOC;                               % We consider a case where vehicles are evenly distributed. 

    ori_xy=[col_latitude,col_longitude];
%% ------------------------ Start running the simulation here ------------------------------
    for test_idx = 1:1:NR_TEST                                              % This for loop repeats the experiments for NR_TEST times
        %%
        perturbed_indices = randi([1, NR_REAL_LOC], 1, NR_PER_LOC);
        per_points = [col_latitude(perturbed_indices', 1), col_longitude(perturbed_indices', 1)];
        loss_matrix = pdist2([col_latitude, col_longitude], per_points);
        loss_matrix_max = min(loss_matrix, [], 2);
        [~, ~, ~, cornerPoints_ori, ~, ~, ~, ~] = uniform_anchor(col_latitude, col_longitude, loss_matrix_max, cell_size(1, city_idx));
        close all;

        allPoints = cornerPoints_ori;
        for layer_id=1:1:3
            cornerPoints = allPoints;
            interpolation_extension_k;
            distMat = pdist2(allPoints, per_points);
            loss_matrix_selected = distMat;  
            for epsilon_idx = 1:1:EPSILON_MAX
                EPSILON = 0.5 * epsilon_idx;
                
    
              
                t_em_start = tic;
                P_matrix=zeros(size(loss_matrix_selected,1),size(loss_matrix_selected,2));
                sum_i=zeros(size(loss_matrix_selected,1),1);
                for i=1:size(loss_matrix_selected,1)
                    for j=1:size(loss_matrix_selected,2)
                        sum_i(i,1)=sum_i(i,1)+exp(-EPSILON*distMat(i,j)/2.0);
                    end
                    for j=1:size(loss_matrix_selected,2)
                        P_matrix(i,j)=exp(-EPSILON*distMat(i,j)/2.0)/sum_i(i,1);
                    end
                end
                cost = loss_matrix_selected/size(loss_matrix_selected,1);
                loss_EM = sum(sum(cost .* P_matrix));
    
    
                % P_2=zeros(size(loss_matrix_selected,2),size(loss_matrix_selected,1));
                % for i=1:size(loss_matrix_selected,2)
                %     for j=1:size(loss_matrix_selected,1)
                %         P_2(i,j)=P_matrix(j,i)/sum(P_matrix(:,i));
                %     end
                % end
                % y_k=sparse(size(loss_matrix_selected,1),0);
                % for i=1:size(loss_matrix_selected,2)
                %     sum_pc=[];
                %     for j=1:size(loss_matrix_selected,2)
                %         sum_pc_j=P_2(i,:)*cost(:,j);
                %         sum_pc=[sum_pc,sum_pc_j];
                %     end
                %     [min_sum, y_k(i)] = min(sum_pc);
                % end
                % 
                % BR_loss=sum(sum(cost(:,y_k) .* P_matrix));
                prior_br=ones(1, size(P_matrix,1))/size(P_matrix,1);
                BR_loss=perturbation_cal_rmp(P_matrix, prior_br, loss_matrix_selected);
                time_em_ep(epsilon_idx+3*layer_id-3) = toc(t_em_start);

                loss_em(epsilon_idx+3*layer_id-3)   = loss_EM;
                loss_embr(epsilon_idx+3*layer_id-3) = BR_loss;

            end
        end
        all_em(test_idx, :)   = loss_em;
        all_embr(test_idx, :) = loss_embr;
        all_time_em(test_idx, :) = time_em_ep;
    end
end
mean_em   = mean(all_em,   1);
std_em    = std(all_em,    0, 1);
mean_embr = mean(all_embr, 1);
std_embr  = std(all_embr,  0, 1);
parts_em   = cell(1, 9);
parts_embr = cell(1, 9);
for k = 1:9
    parts_em{k}   = sprintf('%.2f%s%.2f', mean_em(k),   char(177), std_em(k));
    parts_embr{k} = sprintf('%.2f%s%.2f', mean_embr(k), char(177), std_embr(k));
end
fprintf('%s\n', strjoin(parts_em,   ' & '));
fprintf('%s\n', strjoin(parts_embr, ' & '));

mean_time_em = mean(all_time_em, 1);
std_time_em  = std(all_time_em, 0, 1);
parts_t_em   = cell(1, 9);
parts_t_embr = cell(1, 9);
for k = 1:9
    parts_t_em{k}   = sprintf('%.2f%s%.2f', mean_time_em(k), char(177), std_time_em(k));
    parts_t_embr{k} = sprintf('%.2f%s%.2f', mean_time_em(k), char(177), std_time_em(k));
end
fprintf('EM [%s] time: %s\n',   city, strjoin(parts_t_em,   ' & '));
fprintf('EMBR [%s] time: %s\n', city, strjoin(parts_t_embr, ' & '));
fid = fopen('results_time.txt', 'a');
fprintf(fid, 'EM [%s]: %s\n',   city, strjoin(parts_t_em,   ' & '));
fprintf(fid, 'EMBR [%s]: %s\n', city, strjoin(parts_t_embr, ' & '));
fclose(fid);
