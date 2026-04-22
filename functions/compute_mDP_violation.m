function [violation_ratio, log_ratio] = compute_mDP_violation(Z, D, epsilon)
% compute_mDP_violation - Calculate mDP violation ratio
%
% Syntax: violation_ratio = compute_mDP_violation(Z, D, epsilon)
%
% Inputs:
%    Z - Perturbation matrix (n_x by n_y), where Z(i,k) = P(y_k | x_i)
%    D - Distance matrix (n_x by n_x), where D(i,j) = distance between x_i and x_j
%    epsilon - Privacy budget
%
% Output:
%    violation_ratio - Fraction of (i,j,k) that violate the (epsilon, d_p)-mDP constraint

% Size
[n_x, n_y] = size(Z);

% Initialize violation counter
violation_count = 0;
total_count = 0;
% log_ratio = 0; 
log_ratio = zeros(1, 1, n_x*(n_x-1)*n_y);

% Loop over all pairs (i, j) and outputs k
for i = 1:n_x
    for j = 1:n_x
        if i == j
            continue; % Skip same points (distance zero, meaningless)
        end
        for k = 1:n_y
            
            % Avoid division by zero
            if Z(i, k) - Z(j, k)*exp(epsilon * D(i, j)) - 0.01 > 0
                violation_count = violation_count + 1;
            end
            if Z(j, k) - Z(i, k)*exp(epsilon * D(i, j)) - 0.01 > 0
                violation_count = violation_count + 1;
            end
            % log_ratio = abs(log(Z(i, k) / Z(j, k)));
            % log_ratio_sum = log_ratio_sum+log_ratio/(epsilon * D(i, j));
            % if Z(j, k) > 0
            % 
            %     threshold = epsilon * D(i, j)+0.01;
            %     
            %     if abs(log_ratio) > threshold
            %         violation_count = violation_count + 1;
            %     end
            % elseif Z(i, k) > 0
            %     % If Z(j,k)=0 but Z(i,k)>0, log ratio is +inf -> violation
            %     violation_count = violation_count + 1;
            % end
            total_count = total_count + 1;
            log_ratio(1, total_count) = min(10000, abs(log((Z(i, k)+0.001)/(Z(j, k)+0.001)))/D(i,j));
        end
    end
end

% Compute violation ratio
violation_ratio = violation_count / total_count;

end
