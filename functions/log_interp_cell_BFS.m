function [x_opt, cost] = log_interp_cell_BFS(x_anchor_opt, corner_weights_selected, loss_matrix_max, prior)
% logconv_interp_cellwise: log-convex interpolation processed cell by cell.
% For each cell (defined by 4 anchor corners), interpolate all locations
% that belong to that cell.
%
% Inputs:
%   x_anchor_opt           : nAnchors x nPerturb, perturbation matrix on anchors
%   corner_weights_selected: nLoc x nAnchors, bilinear weights (sparse, 4 non-zeros per row)
%   loss_matrix_max        : nLoc x nPerturb, loss matrix for each location
%
% Outputs:
%   x_opt : nLoc x nPerturb, interpolated perturbation matrix
%   cost  : scalar, average utility loss

    [nr_loc, nr_perturb] = size(loss_matrix_max);
    x_opt = zeros(nr_loc, nr_perturb);

    % Identify all unique cells.
    % Each cell is defined by a unique set of 4 anchor indices (non-zero columns per row).
    % We group locations by their cell (i.e., by their 4 corner anchor indices).
    W = corner_weights_selected;  % nLoc x nAnchors

    % Build a cell signature for each location: sorted non-zero column indices
    cell_signatures = zeros(nr_loc, 4);
    for i = 1:nr_loc
        nz = find(W(i, :) > 0);
        if length(nz) == 4
            cell_signatures(i, :) = sort(nz);
        else
            % Degenerate case (boundary/corner point): pad with zeros
            sig = zeros(1, 4);
            sig(1:length(nz)) = sort(nz);
            cell_signatures(i, :) = sig;
        end
    end

    % Find unique cells
    [unique_cells, ~, cell_id] = unique(cell_signatures, 'rows');
    nr_cells = size(unique_cells, 1);

    % Process cell by cell
    for c = 1:nr_cells
        anchor_idx = unique_cells(c, :);
        anchor_idx = anchor_idx(anchor_idx > 0);  % remove padding zeros

        % All locations belonging to this cell
        loc_in_cell = find(cell_id == c);

        for i = loc_in_cell'
            anchor_w = W(i, anchor_idx);  % weights for this location's anchors

            % Log-convex interpolation in log space
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
    end

    if nargin < 4 || isempty(prior)
        cost = sum(sum(x_opt .* loss_matrix_max)) / nr_loc;
    else
        cost = prior' * sum(x_opt .* loss_matrix_max, 2);
    end
end