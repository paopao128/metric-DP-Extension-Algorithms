function [x_opt, cost] = log_interp_loc_BFS(x_anchor_opt, corner_weights_selected, loss_matrix_max)
% logconv_interp_cellwise: log-convex interpolation processed cell by cell.
% For each output location i, only the (up to 4) non-zero corner weights
% contribute, corresponding to the cell that contains location i.
%
% Inputs:
%   x_anchor_opt          : nAnchors x nPerturb, perturbation matrix on anchors
%   corner_weights_selected: nLoc x nAnchors, bilinear weights (sparse, 4 non-zeros per row)
%   loss_matrix_max       : nLoc x nPerturb, loss matrix for each location
%
% Outputs:
%   x_opt : nLoc x nPerturb, interpolated perturbation matrix
%   cost  : scalar, average utility loss

    [nr_loc, nr_perturb] = size(loss_matrix_max);
    x_opt = zeros(nr_loc, nr_perturb);

    for i = 1:nr_loc
        % Find the non-zero anchor indices and weights for this location
        % (corresponds to the 4 corners of the cell containing location i)
        w = corner_weights_selected(i, :);       % 1 x nAnchors
        anchor_idx = find(w > 0);                % indices of contributing anchors (up to 4)
        anchor_w   = w(anchor_idx);              % their weights, sum = 1

        % Log-convex interpolation: geometric weighted average in probability space
        % log(x_opt(i,k)) = sum_j( anchor_w(j) * log(x_anchor_opt(anchor_idx(j), k)) )
        log_z = zeros(1, nr_perturb);
        for j = 1:length(anchor_idx)
            log_z = log_z + anchor_w(j) * log(x_anchor_opt(anchor_idx(j), :) + eps);
        end
        z = exp(log_z);

        % Normalize
        z_sum = sum(z);
        if ~isnan(z_sum) && z_sum > 0
            x_opt(i, :) = z / z_sum;
        else
            x_opt(i, :) = 0;
        end
    end

    cost = sum(sum(x_opt .* loss_matrix_max)) / nr_loc;
end