function [x, fval] = perturbation_cal_em(selected_longitudes, selected_latitudes, perturbed_longitudes, perturbed_latitudes, loss_matrix_selected, EPSILON)
    nr_node = size(selected_longitudes, 1); 
    nr_perturb = size(perturbed_longitudes, 1);
    x = zeros(nr_node, nr_perturb); 
    for i = 1:1:nr_node
        for k = 1:1:nr_perturb
            distance_matrix(i,k) = norm([selected_latitudes(i, 1), selected_longitudes(i, 1)] - [perturbed_latitudes(k, 1), perturbed_longitudes(k, 1)], 2); 
        end
        x(i, :) = exp(-distance_matrix(i, :)*EPSILON/2); 
        x(i, :) = x(i, :)/sum(x(i, :)); 
    end

    fval =sum(sum(x.*loss_matrix_selected))/nr_node; 

    % coords = [selected_latitudes(:), selected_longitudes(:)];
    % total_count = 0;
    % violation_count = 0;
    % for i = 1:1000
    %     for j = 1:1000
    %         if i == j
    %             continue; % Skip same points (distance zero, meaningless)
    %         end
    %         for k = 1:20
    %             % Avoid division by zero
    %             idx_i = samples_viocheck(1,i); 
    %             idx_j = samples_viocheck(1,j); 
    %             diff = coords(idx_i,:) - coords(idx_j,:);
    %             distance_1 = norm(diff, 2);
    %             distance_2 = euclidean_distance_matrix_viocheck(i,j); 
    %             if x(idx_i, k) - x(idx_j, k)*exp(EPSILON * euclidean_distance_matrix_viocheck(i,j)) > 0
    %                 violation_count = violation_count + 1;
    %             end
    %             if x(idx_j, k) - x(idx_i, k)*exp(EPSILON * euclidean_distance_matrix_viocheck(i,j)) > 0
    %                 violation_count = violation_count + 1;
    %             end
    %             total_count = total_count + 1;
    %         end
    %     end
    % end
    % violation_ratio = violation_count / total_count;

end