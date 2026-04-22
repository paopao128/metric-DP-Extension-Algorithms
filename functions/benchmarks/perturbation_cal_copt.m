function [H, M, expected_loss] = perturbation_cal_copt(D_rp, D_rr, loss_matrix, grid_prior, epsilon, lambda, r)
    [n, m] = size(D_rp);  % n: real locations, m: perturbed locations
    N = n * m;
    Y_start = N + 1;
    Y_end = Y_start + m - 1;
    k_idx = Y_end + 1;
    total_vars = k_idx;

    %% Build r-nearest neighbors I(v) over real locations
    I = cell(m, 1);
    for v = 1:m
        [~, idx] = sort(D_rp(:, v));
        I{v} = idx(1:r);
    end

    %% LP inequality constraints: A * x ≤ b
    rows = []; cols = []; vals = []; b = [];
    row_count = 0;

    % 1. Utility constraint: tilde L(M, w) ≤ k
    for u = 1:n
        row_count = row_count + 1;
        for v = 1:m
            idx = sub2ind([n, m], u, v);
            if ismember(u, I{v})
                coeff = loss_matrix(u, v) + lambda;
                rows(end+1) = row_count; 
                cols(end+1) = idx; 
                vals(end+1) = coeff;
               
            else
                y_idx = Y_start + v - 1;
                coeff = loss_matrix(u, v) * exp(-epsilon * D_rp(u, v));
                rows(end+1) = row_count; 
                cols(end+1) = y_idx; 
                vals(end+1) = coeff;
                
            end
        end
        rows(end+1) = row_count; 
        cols(end+1) = k_idx; 
        vals(end+1) = -1;
        b(end+1, 1) = 0;
    end

    % 2. Row sum constraint: sum ≥ 1 → -sum ≤ -1
    for u = 1:n
        row_count = row_count + 1;
        for v = 1:m
            idx = sub2ind([n, m], u, v);
            if ismember(u, I{v})
                rows(end+1) = row_count; 
                cols(end+1) = idx; 
                vals(end+1) = -1;
            else
                y_idx = Y_start + v - 1;
                rows(end+1) = row_count; 
                cols(end+1) = y_idx; 
                vals(end+1) = -min(exp(epsilon * D_rr(u, v)), 1e10);

            end
        end
        b(end+1, 1) = -1;
    end

    % 3. mDP constraints: M_uw ≤ M_vw * exp(ε ⋅ d(u,v))
    for w = 1:m
        Iw = I{w};  % Only enforce for u in I{w}
        for u_idx = 1:length(Iw)
            u = Iw(u_idx);
            for v_idx = 1:1:length(Iw)
                v = Iw(v_idx);
                if u ~= v
                    row_count = row_count + 1;
                    idx_uw = sub2ind([n, m], u, w);
                    idx_vw = sub2ind([n, m], v, w);
                    rows(end+1) = row_count; 
                    cols(end+1) = idx_uw; 
                    vals(end+1) = 1;
                    rows(end+1) = row_count; 
                    cols(end+1) = idx_vw; 
                    vals(end+1) = -min(exp(epsilon * D_rr(u, v)), 1e10);
                    b(end+1, 1) = 0;
                end
            end
        end
    end

    A = sparse(rows, cols, vals, row_count, total_vars);

    %% 4. Equality constraints: M(u, v) = Y(v) * exp(-ε d(u, v)) for u ∉ I(v)
    Aeq_rows = []; Aeq_cols = []; Aeq_vals = []; beq = []; row_eq = 0;

    for v = 1:m
        u_all = setdiff(1:n, I{v});
        for u = u_all
            % if exp(epsilon * D_rp(u, v)) < 1e4
                row_eq = row_eq + 1;
                idx_uv = sub2ind([n, m], u, v);
                idx_yv = Y_start + v - 1;

                Aeq_rows(end+1) = row_eq;
                Aeq_cols(end+1) = idx_uv;
                Aeq_vals(end+1) = 1;

                Aeq_rows(end+1) = row_eq;
                Aeq_cols(end+1) = idx_yv;
                Aeq_vals(end+1) = -exp(-epsilon * D_rp(u, v));

                % Aeq_vals(end+1) = min(-exp(-epsilon * D_rp(u, v)), -1/1e10); 
                beq(end+1, 1) = 0;
            % end
        end
    end

    Aeq = sparse(Aeq_rows, Aeq_cols, Aeq_vals, row_eq, total_vars);

    %% Objective: minimize k
    f = zeros(total_vars, 1); f(k_idx) = 1;
    lb = ones(total_vars, 1)*0.0000000001;
    ub = ones(total_vars, 1);
    %% Solve LP
    options = optimoptions('linprog', 'Display', 'off', 'Algorithm', 'dual-simplex');
    [x, fval, exitflag] = linprog(f, A, b, Aeq, beq, lb, [], options);

    if exitflag ~= 1
        error('LP did not converge.');
    end

    %% Reconstruct full M matrix
    M_opt = reshape(x(1:N), n, m);
    Y = x(Y_start:Y_end);

    M = zeros(n, m);
    for u = 1:n
        for v = 1:m
            if ismember(u, I{v})
                M(u, v) = M_opt(u, v);
            else
                M(u, v) = Y(v) * exp(-epsilon * D_rp(u, v));
            end
        end
    end

    %% Normalize rows of M to get stochastic matrix H
    H = zeros(n, m);
    for u = 1:n
        row_sum = sum(M(u, :));
        if row_sum > 0
            H(u, :) = M(u, :) / row_sum;
        else
            H(u, :) = ones(1, m) / m;
        end
    end

    %% Compute worst-case expected utility loss
    expected_loss = 0;
    % for u = 1:n
    %     loss = sum(H(u, :) .* loss_matrix(u, :));
    %     expected_loss = max(expected_loss, loss);
    % end
    expected_loss = sum(sum((grid_prior'.*H).*loss_matrix)); 
end
