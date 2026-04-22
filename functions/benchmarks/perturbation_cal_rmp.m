function QL = perturbation_cal_rmp(K, pi, D)
% BAYESIAN_REMAP_EXPECTED_LOSS
%   Computes expected utility loss after Bayesian remap
%
% Inputs:
%   K  - (n x m) perturbation matrix: P(z | x)
%   pi - (1 x n) prior distribution over true locations
%   D  - (n x m) cost matrix: D(x, z) = loss when reporting z for true x
%
% Output:
%   QL - scalar: overall expected utility loss after remapping

    [n, m] = size(K);
    
    if size(pi, 1) ~= 1
        pi = pi'; % Ensure prior is a row vector
    end

    QL = 0;

    for z = 1:m
        % --- Compute posterior sigma(x | z) ∝ pi(x) * K(x, z)
        posterior_numerator = pi .* K(:, z)';
        posterior_denominator = sum(posterior_numerator);
        
        if posterior_denominator == 0
            continue;  % skip if z is never reported
        end
        
        sigma = posterior_numerator / posterior_denominator;  % posterior over x

        % --- Find best remap location z_star minimizing expected loss
        expected_losses = sigma * D;  % (1 x m) vector of expected loss for each z_star
        [~, z_star] = min(expected_losses);  % index of optimal remap location

        % --- Add to total expected loss
        for x = 1:n
            QL = QL + pi(x) * K(x, z) * D(x, z_star);
        end
    end
end
