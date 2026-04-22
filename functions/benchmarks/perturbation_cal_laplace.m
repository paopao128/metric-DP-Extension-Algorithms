function [K, QL] = perturbation_cal_laplace(loc_lons, loc_lats, pert_lons, pert_lats, loss_matrix, epsilon)
% PLANAR_LAPLACE_UTILITY_LOSS   ε-geo-indistinguishable mechanism + utility loss
%
% Inputs:
%   loc_lons, loc_lats     n×1 true coordinates
%   pert_lons, pert_lats   m×1 grid points
%   loss_matrix            n×m utility loss matrix
%   epsilon                desired privacy budget
%   u                      grid spacing
%   r_max                  maximum support radius
%   delta_theta            angular resolution (radians)
%
% Outputs:
%   K   n×m mechanism matrix (rows sum to 1)
%   QL  scalar expected utility loss under uniform prior

    % 1) Euclidean distance matrix
    u = log(3/2)/epsilon; 
    loc  = [loc_lons(:),  loc_lats(:)];
    pert = [pert_lons(:), pert_lats(:)];
    D    = pdist2(loc, pert, 'euclidean');  % n×m

    % 2) Compute ε′ via Theorem 4.1 if valid
    delta_theta = 1e-6; 
    r_max = u/(3*delta_theta);
    u = log(1.5)/epsilon;
    q = 3;
    fudge_needed = (r_max < u / delta_theta) && (q > 2);

    if fudge_needed
        % Fudge function from Theorem 4.1
        fudge = @(ep) ep + (1/u)*log((q + 2*exp(ep*u)) ./ (q - 2*exp(ep*u))) - epsilon;
        
        % Safe upper bracket
        ep_max = min(epsilon, (1/u)*log(q/2) * 0.999);
        
        % Check if the function crosses zero
        f_lo = fudge(0);
        f_hi = fudge(ep_max);
        
        if f_lo * f_hi > 0
            % Both values same sign ⇒ fallback
            % warning('fzero bracket invalid: fallback to ε′ = ε');
            eps_prime = epsilon/2;
        else
            % OK to solve
            eps_prime = fzero(fudge, [0, ep_max], optimset('Display','off'))/2;
        end
    else
        % Otherwise: fall back to exact Laplace kernel with ε′ = ε
        eps_prime = epsilon;
    end

    % 3) Construct kernel
    P = exp(-(eps_prime) * D);
    K = P ./ sum(P,2);

    % 4) Expected utility loss under uniform prior
    n = size(D,1);
    QL = sum(sum(K .* loss_matrix, 2)) / n;
end
