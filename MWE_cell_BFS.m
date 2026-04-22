function [M_tilde, loss] = MWE_cell_BFS(cornerPoints, z_anchor_instance, allPoints, eta, loss_matrix_selected, prior)
% McShane-Whitney Extension in log space, cell-wise version.
% For each new point, only the 4 anchor corners of its containing cell
% are used to compute the McShane-Whitney envelopes.
%
% Inputs:
%   cornerPoints        - nAnchors x 2, anchor point coordinates
%   z_anchor_instance   - nAnchors x nY, probability distributions on anchors M(y|a)
%   allPoints           - nAll x 2, all points coordinates (includes anchors)
%   eta                 - Lipschitz budget parameter (default: epsilon/2)
%   loss_matrix_selected- nAll x nY, loss matrix for all points
%
% Output:
%   M_tilde             - nAll x nY, extended probability distributions
%   loss                - scalar, average utility loss

    if nargin < 4
        eta = 0.5;
    end

    num_all    = size(allPoints, 1);
    num_y      = size(z_anchor_instance, 2);

    M_tilde = zeros(num_all, num_y);

    % -------------------------------------------------------
    % Build the anchor grid structure to identify cells
    % -------------------------------------------------------
    xU = sort(unique(cornerPoints(:, 1)));
    yU = sort(unique(cornerPoints(:, 2)));
    nU = length(yU);
    mU = length(xU);

    % cpGrid(row, col) = index into cornerPoints
    cpGrid = zeros(nU, mU);
    for t = 1:size(cornerPoints, 1)
        col_g = find(xU == cornerPoints(t, 1));
        row_g = find(yU == cornerPoints(t, 2));
        cpGrid(row_g, col_g) = t;
    end

    % -------------------------------------------------------
    % For each point in allPoints, find its cell and compute MWE
    % using only the 4 corners of that cell
    % -------------------------------------------------------
    for i = 1:num_all
        px = allPoints(i, 1);
        py = allPoints(i, 2);

        % Find the cell containing this point
        cx = find(xU <= px, 1, 'last');
        cy = find(yU <= py, 1, 'last');

        % Clamp to valid cell range
        if isempty(cx) || cx >= mU, cx = max(1, mU - 1); end
        if isempty(cy) || cy >= nU, cy = max(1, nU - 1); end

        % 4 corner indices of this cell
        i00 = cpGrid(cy,   cx);
        i10 = cpGrid(cy,   cx+1);
        i01 = cpGrid(cy+1, cx);
        i11 = cpGrid(cy+1, cx+1);
        cell_anchors = [i00, i10, i01, i11];

        % Anchor coords and log-probs for this cell only
        cell_coords = cornerPoints(cell_anchors, :);   % 4 x 2
        g_cell      = log(z_anchor_instance(cell_anchors, :) + eps);  % 4 x nY

        % Distance from this point to its 4 cell corners
        diff_x = px - cell_coords(:, 1);   % 4 x 1
        diff_y = py - cell_coords(:, 2);   % 4 x 1
        dist   = sqrt(diff_x.^2 + diff_y.^2);  % 4 x 1

        eta_dist = eta * dist;  % 4 x 1

        % McShane-Whitney envelopes (over 4 anchors only)
        l_y = max(g_cell - eta_dist, [], 1);   % 1 x nY  lower envelope
        u_y = min(g_cell + eta_dist, [], 1);   % 1 x nY  upper envelope

        % Midpoint
        g_tilde = (l_y + u_y) / 2;  % 1 x nY

        % Exponentiate and normalize
        w = exp(g_tilde);
        w_sum = sum(w);
        if w_sum > 0 && ~isnan(w_sum)
            M_tilde(i, :) = w / w_sum;
        else
            M_tilde(i, :) = 0;
        end
    end

    % -------------------------------------------------------
    % Compute utility loss
    % -------------------------------------------------------
    if nargin < 6 || isempty(prior)
        loss = sum(sum(M_tilde .* loss_matrix_selected)) / size(loss_matrix_selected, 1);
    else
        loss = prior' * sum(M_tilde .* loss_matrix_selected, 2);
    end
end