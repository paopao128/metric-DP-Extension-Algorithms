function [K, QL] = perturbation_cal_tem(loc_lons, loc_lats, pert_lons, pert_lats, D, epsilon)
% TEM_METRIC_DP
%   Implements the Metric Truncated Exponential Mechanism (TEM) over geographic coordinates.
%
% Inputs:
%   loc_lons, loc_lats     - true location coordinates (length n)
%   pert_lons, pert_lats   - candidate output locations (length m)
%   D                      - cost matrix: D(i,j) = cost from true i to perturbed j
%   epsilon                - privacy parameter (ε)
%
% Outputs:
%   K                      - perturbation matrix: K(i,j) = Pr[output j | true i]
%   QL                     - expected utility loss (scalar)

    n = length(loc_lons);    % number of true locations
    m = length(pert_lons);   % number of output locations
    R = 6371;                % Earth radius in km
    loc_lons = loc_lons/100;
    loc_lats = loc_lats/100;
    epsilon = epsilon*100; 
    % Step 1: Compute distance matrix (n x m)
    dist_mat = zeros(n, m);
    for i = 1:n
        for j = 1:m
            % dist_mat(i, j) = haversine(loc_lons(i), loc_lats(i), pert_lons(j), pert_lats(j), R);
            dist_mat(i, j) = norm([loc_lons(i, 1), loc_lats(i, 1)] - [pert_lons(j, 1), pert_lats(j, 1)]);
        end
    end

    % Step 2: Set truncation threshold γ (based on Theorem 4.2)
    beta = 3;  % probability mass allowed outside γ
    gamma = (2 / epsilon) * log((m - 1) * (1 - beta) / beta);

    % Step 3: Initialize perturbation matrix K
    K = zeros(n, m);

    for i = 1:n
        % Indices of nearby elements within γ
        L = find(dist_mat(i,:) <= gamma);
        Lc = find(dist_mat(i,:) > gamma);  % complement (outside γ)

        % Score function: f(i,j) = -d(i,j), clipped at γ
        scores = -dist_mat(i,:);
        scores(Lc) = -gamma;

        % Add Gumbel noise to all scores
        gumbel_noise = random_gumbel(1, m) * (0.001 / epsilon);
        noisy_scores = scores + gumbel_noise;

        % Apply softmax over noisy scores to get probabilities
        probs = exp(noisy_scores - max(noisy_scores));  % for numerical stability
        probs = probs / sum(probs);

        K(i,:) = probs;
    end

    % Step 4: Compute expected utility loss with uniform prior
    pi_prior = ones(1, n) / n;
    QL = real(sum(pi_prior * (K .* D)));
end

function d = haversine(lon1, lat1, lon2, lat2, R)
    dlat = deg2rad(lat2 - lat1);
    dlon = deg2rad(lon2 - lon1);
    a = sin(dlat/2).^2 + cos(deg2rad(lat1)) .* cos(deg2rad(lat2)) .* sin(dlon/2).^2;
    c = 2 * atan2(sqrt(a), sqrt(1 - a));
    d = R * c;
end

function g = random_gumbel(m, n)
% Generate Gumbel(0,1) noise matrix of size m x n
    U = rand(m, n);
    g = -log(-log(U));
end
