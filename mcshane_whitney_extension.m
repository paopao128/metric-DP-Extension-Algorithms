function [M_tilde,loss] = mcshane_whitney_extension(cornerPoints, z_anchor_instance, allPoints, eta, loss_matrix_selected, prior)
% McShane-Whitney Extension in log space
%
% Inputs:
%   cornerPoints        - 132x2, anchor points coordinates
%   z_anchor_instance   - 132x20, probability distributions on anchors M(y|a)
%   allPoints           - 483x2, all points coordinates (includes anchors)
%   eta                 - Lipschitz budget parameter (default: epsilon/2)
%
% Output:
%   M_tilde             - 483x20, extended probability distributions

    if nargin < 4
        eta = 0.5;  % default eta, adjust as needed
    end

    num_all    = size(allPoints, 1);   % 483
    num_anchor = size(cornerPoints, 1); % 132
    num_y      = size(z_anchor_instance, 2); % 20

    % -------------------------------------------------------
    % Step 1: Compute log-probabilities on anchors
    % g_y(a) = log M(y | a),  shape: 132 x 20
    % -------------------------------------------------------
    g_anchor = log(z_anchor_instance);  % 132 x 20

    % -------------------------------------------------------
    % Step 2: Compute pairwise distances d(x, a)
    % dist(i, j) = d(allPoints_i, cornerPoints_j)
    % shape: 483 x 132
    % -------------------------------------------------------
    % Using Euclidean distance
    diff_x = allPoints(:,1) - cornerPoints(:,1)';   % 483 x 132
    diff_y = allPoints(:,2) - cornerPoints(:,2)';   % 483 x 132
    dist   = sqrt(diff_x.^2 + diff_y.^2);           % 483 x 132

    % -------------------------------------------------------
    % Step 3: Compute McShane-Whitney envelopes for each y
    % l_y(x) = sup_a [ g_y(a) - eta * d(x,a) ]  (lower envelope)
    % u_y(x) = inf_a [ g_y(a) + eta * d(x,a) ]  (upper envelope)
    % shape: 483 x 20
    % -------------------------------------------------------
    % eta_dist(i,j) = eta * d(allPoints_i, anchor_j), shape: 483 x 132
    eta_dist = eta * dist;

    l_y = zeros(num_all, num_y);
    u_y = zeros(num_all, num_y);

    for k = 1:num_y
        g_k = g_anchor(:, k)';  % 1 x 132, g_y(a) for output k

        % lower envelope: max over anchors of [g_y(a) - eta*d(x,a)]
        l_y(:, k) = max(g_k - eta_dist, [], 2);  % 483 x 1

        % upper envelope: min over anchors of [g_y(a) + eta*d(x,a)]
        u_y(:, k) = min(g_k + eta_dist, [], 2);  % 483 x 1
    end

    % -------------------------------------------------------
    % Step 4: Choose g_tilde as midpoint, shape: 483 x 20
    % -------------------------------------------------------
    g_tilde = (l_y + u_y) / 2;

    % -------------------------------------------------------
    % Step 5: Exponentiate and normalize to get probability
    % M_tilde(y|x) = exp(g_tilde_y(x)) / sum_{y'} exp(g_tilde_{y'}(x))
    % -------------------------------------------------------
    w_tilde = exp(g_tilde);                          % 483 x 20
    w_sum   = sum(w_tilde, 2);                       % 483 x 1
    M_tilde = w_tilde ./ w_sum;                      % 483 x 20

    % -------------------------------------------------------
    % Sanity check: verify anchor rows match original
    % -------------------------------------------------------
    % (Optional) Find which rows of allPoints correspond to anchors
    % and verify consistency
    if nargin < 6 || isempty(prior)
        loss = sum(sum(M_tilde .* loss_matrix_selected)) / size(loss_matrix_selected, 1);
    else
        loss = prior' * sum(M_tilde .* loss_matrix_selected, 2);
    end
end