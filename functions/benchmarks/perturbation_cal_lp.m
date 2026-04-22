function [p_opt, cost] = perturbation_cal_lp(grid_utility_loss, grid_distances, grid_prior, epsilon)
    % grid_utility_loss: [n_input x m_output]
    % grid_distances: [n_input x n_input]
    [n, m] = size(grid_utility_loss);  % n: input cells, m: output cells
    num_vars = n * m;

    % Objective: minimize expected utility loss
    f = grid_utility_loss(:);  % flatten to (n*m x 1)

    % === Equality constraints: each row (input) must sum to 1 ===
    Aeq = sparse(n, num_vars);
    beq = ones(n, 1);
    for i = 1:n
        for j = 1:m
            var_idx = sub2ind([n, m], i, j);
            Aeq(i, var_idx) = 1;
        end
    end

    % === Inequality constraints: geo-indistinguishability for all i ≠ k, ∀j ===
    rows = [];
    cols = [];
    vals = [];
    b = [];

    constraint_counter = 0;

    for j = 1:m
        for i = 1:n
            for k = 1:n
                if i == k
                    continue;
                end
                constraint_counter = constraint_counter + 1;
                idx_ij = sub2ind([n, m], i, j);
                idx_kj = sub2ind([n, m], k, j);
                alpha = min([exp(epsilon * grid_distances(i, k)), 10e6]);

                rows(end+1) = constraint_counter;
                cols(end+1) = idx_ij;
                vals(end+1) = 1;

                rows(end+1) = constraint_counter;
                cols(end+1) = idx_kj;
                vals(end+1) = -alpha;

                b(end+1, 1) = 0;
            end
        end
    end

    % Sparse inequality matrix
    A = sparse(rows, cols, vals, constraint_counter, num_vars);

    % Variable bounds
    lb = zeros(num_vars, 1);
    ub = ones(num_vars, 1);  % optional

    % Solve LP
    options = optimoptions('linprog', 'Display', 'off');
    % options = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'off');

    [p_vec, fval, exitflag] = linprog(f, A, b, Aeq, beq, lb, ub, options);

    if exitflag ~= 1
        p_opt = ones(n, m)/m; 
    else
        p_opt = reshape(p_vec, [n, m]);
    end

    % Reshape result: p(i,j)
    % p_opt = reshape(p_vec, [n, m]);
    cost = sum(sum((grid_prior'.*p_opt).*grid_utility_loss));
end
