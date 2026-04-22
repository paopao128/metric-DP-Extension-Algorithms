function [x_opt, fval_inst] = perturbation_cal_apo(c, lambda, d, neighborMatrix, epsilon1, epsilon2)
    [nr_anchor, nr_perturb] = size(c); 
    for i = 1:1:nr_anchor
        for k = 1:1:nr_perturb
            %f((k-1)*nr_anchor + i) = max([c(i, k), 999999]); 
            f((k-1)*nr_anchor + i) = min([c(i, k), 999999]);
        end
    end
    % Dimensions
    [A, K] = size(c);       % c: (A x K)
    J = size(lambda, 2);    % lambda: (A x J)


    Aeq = []; beq = [];
    Aineq = []; bineq = [];
    lb = ones(J*K,1)*0.000001;  % positivity
    ub = ones(J*K,1);            % no direct upper bound except for constraints

    % (a) sum_k x_{i,k} = 1
    % Build Aeq: size = J x (J*K)
    Aeq = buildSumToOne(J, K);  % You must implement this
    beq = ones(J,1);

    % (b) x_{j,k} - exp(epsilon*d(i,j)) * x_{i,k} <= 0
    % [Aineq, bineq] = buildExpConstraints(J, K, d, epsilon);  % Implement details
    [Aineq, bineq] = buildDirectionalExpConstraints(J, K, d, neighborMatrix, epsilon1, epsilon2); 
    % We'll solve:
    %    min c_obj' * xVec   (plus optional constant offset c_cons)
    % subject to Aeq*xVec=beq, Aineq*xVec<=bineq, lb<=xVec<=ub

    options = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'off');
    [xVec, fval_inst, exitflag] = linprog(f, Aineq, bineq, Aeq, beq, lb, ub, options);
    if exitflag ~= 1
        x_opt = ones(J, K)/K; 
    else
        x_opt = reshape(xVec, [J,K]);
    end
end


% -------------- Utility Functions --------------


function Aeq = buildSumToOne(J, K)
    % We want sum_{k} x_{i,k} = 1 for each i in 1..J.
    % So for i-th row, we have x_{i,1} + x_{i,2} + ... + x_{i,K} = 1
    % The row i of Aeq has 1’s in columns corresponding to x_{i,*}.
    Aeq = zeros(J, J*K);
    for i = 1:J
        for k = 1:K
            col = sub2ind([J,K], i, k);
            Aeq(i,col) = 1;
        end
    end
end

function [Aineq, bineq] = buildExpConstraints(J, K, d, epsilon)
    % Build sparse constraints only for nonzero entries in d
    % Constraints: x_{j,k} - exp(epsilon * d(i,j)) * x_{i,k} <= 0
    
    [i_idx, j_idx, d_values] = find(d);  % Extract nonzero indices and values from sparse d
    numNonZero = length(i_idx);  % Only process nonzero (i,j) pairs
    numConstraints = numNonZero * K;  % Total constraints
    
    rowIdx = zeros(2 * numConstraints, 1);
    colIdx = zeros(2 * numConstraints, 1);
    values = zeros(2 * numConstraints, 1);
    bineq = zeros(numConstraints, 1);
    
    entry = 1;
    row = 1;
    for idx = 1:numNonZero  % Iterate only over nonzero entries in d
        i = i_idx(idx);
        j = j_idx(idx);
        d_ij = d_values(idx);
        
        expVal = min(exp(epsilon * d_ij), 1e10);  % Prevent overflow
        
        for k = 1:K
            col_jk = sub2ind([J, K], j, k);
            col_ik = sub2ind([J, K], i, k);
            
            % x_{j,k} coefficient: +1
            rowIdx(entry) = row;
            colIdx(entry) = col_jk;
            values(entry) = 1;
            entry = entry + 1;
            
            % x_{i,k} coefficient: -exp(epsilon * d(i,j))
            rowIdx(entry) = row;
            colIdx(entry) = col_ik;
            values(entry) = -expVal;
            entry = entry + 1;
            
            row = row + 1;
        end
    end
    
    % Convert to sparse matrix
    Aineq = sparse(rowIdx, colIdx, values, numConstraints, J * K);
end


function [Aineq, bineq] = buildDirectionalExpConstraints(J, K, distanceMatrix, neighborMatrix, epsilon1, epsilon2)
    % Build sparse constraints using directional epsilon values.
    % Inputs:
    % - J: number of true locations (corners)
    % - K: number of reported locations (corners)
    % - distanceMatrix: sparse matrix of distances between corners
    % - neighborMatrix: sparse matrix where 1 = x-neighbor, 2 = y-neighbor
    % - epsilon1: for x-axis (longitude) neighbors
    % - epsilon2: for y-axis (latitude) neighbors
    
    [i_idx, j_idx, directions] = find(neighborMatrix);
    numConstraints = length(i_idx) * K;

    rowIdx = zeros(2 * numConstraints, 1);
    colIdx = zeros(2 * numConstraints, 1);
    values = zeros(2 * numConstraints, 1);
    % bineq = zeros(numConstraints, 1);
    bineq = ones(numConstraints, 1)*0.0;

    entry = 1;
    row = 1;
    for idx = 1:length(i_idx)
        i = i_idx(idx);
        j = j_idx(idx);
        direction = directions(idx);

        % Use corresponding epsilon based on direction
        if direction == 1
            epsilon = epsilon1;
        elseif direction == 2
            epsilon = epsilon2;
        else
            continue;  % should not happen since we filtered on non-zero neighborMatrix entries
        end

        d_ij = distanceMatrix(i, j);
        %expVal = min(exp(epsilon * d_ij), 1e8);  % Prevent overflow
        expVal = exp(epsilon * d_ij);
        % expVal = exp(epsilon * d_ij); 

        for k = 1:K
            col_jk = sub2ind([J, K], j, k);  % x_{j,k}
            col_ik = sub2ind([J, K], i, k);  % x_{i,k}

            % Constraint: x_{j,k} - exp(eps * d(i,j)) * x_{i,k} <= 0
            rowIdx(entry) = row;
            colIdx(entry) = col_jk;
            values(entry) = 1;
            entry = entry + 1;

            rowIdx(entry) = row;
            colIdx(entry) = col_ik;
            values(entry) = -expVal;
            entry = entry + 1;

            row = row + 1;
        end
    end

    Aineq = sparse(rowIdx, colIdx, values, row - 1, J * K);
end

