function [M_out, loss] = lp_extension_cellwise(cornerPoints, z_anchor, allPoints, epsilon, loss_matrix_selected, prior)
% LP-based cell-wise extension.
% For each cell:
%   - Edge points (non-corner boundary points): bilinear interpolation
%   - Interior points: LP with the 4 corner z values as constants,
%     only interior point distributions as variables.
%
% Inputs:
%   cornerPoints         - nAnchors x 2, anchor (corner) coordinates
%   z_anchor             - nAnchors x nY, known probability distributions on anchors
%   allPoints            - nAll x 2, all points (anchors + new points)
%   epsilon              - privacy budget
%   loss_matrix_selected - nAll x nY, loss matrix for all points
%
% Outputs:
%   M_out - nAll x nY, probability distributions for all points
%   loss  - scalar, average utility loss

    nAll  = size(allPoints, 1);
    nY    = size(z_anchor, 2);
    M_out = zeros(nAll, nY);

    % -------------------------------------------------------
    % Build anchor grid
    % -------------------------------------------------------
    xU = sort(unique(cornerPoints(:, 1)));
    yU = sort(unique(cornerPoints(:, 2)));
    nU = length(yU);
    mU = length(xU);

    cpGrid = zeros(nU, mU);
    for t = 1:size(cornerPoints, 1)
        col_g = find(xU == cornerPoints(t, 1));
        row_g = find(yU == cornerPoints(t, 2));
        cpGrid(row_g, col_g) = t;
    end

    % -------------------------------------------------------
    % For each point in allPoints, record:
    %   - its index into cornerPoints (if it is an anchor), or 0
    %   - its containing cell (cx, cy)
    %   - whether it is a corner, edge, or interior point
    % -------------------------------------------------------
    % First, map allPoints back to anchor indices if applicable
    anchorMap = zeros(nAll, 1);  % anchorMap(i) = anchor index if allPoints(i) is an anchor
    for i = 1:nAll
        for t = 1:size(cornerPoints, 1)
            if norm(allPoints(i,:) - cornerPoints(t,:)) < 1e-10
                anchorMap(i) = t;
                break;
            end
        end
    end

    % Assign corner z values directly
    for i = 1:nAll
        if anchorMap(i) > 0
            M_out(i, :) = z_anchor(anchorMap(i), :);
        end
    end

    % -------------------------------------------------------
    % Process cell by cell
    % -------------------------------------------------------
    for row_g = 1:nU-1
        for col_g = 1:mU-1

            % 4 corner indices into cornerPoints
            i00 = cpGrid(row_g,   col_g);
            i10 = cpGrid(row_g,   col_g+1);
            i01 = cpGrid(row_g+1, col_g);
            i11 = cpGrid(row_g+1, col_g+1);

            x0 = xU(col_g);   x1 = xU(col_g+1);
            y0 = yU(row_g);   y1 = yU(row_g+1);

            % Known z for the 4 corners
            z00 = z_anchor(i00, :);
            z10 = z_anchor(i10, :);
            z01 = z_anchor(i01, :);
            z11 = z_anchor(i11, :);

            % Find all allPoints strictly inside or on edge of this cell
            % (excluding corners)
            inCell = [];  % indices into allPoints
            for i = 1:nAll
                if anchorMap(i) > 0
                    continue;  % skip corners
                end
                px = allPoints(i, 1);
                py = allPoints(i, 2);
                if px >= x0 && px <= x1 && py >= y0 && py <= y1
                    inCell(end+1) = i; %#ok<AGROW>
                end
            end

            if isempty(inCell)
                continue;
            end

            % Classify each point in inCell as edge or interior
            edgeIdx    = [];
            interiorIdx = [];
            for ii = 1:length(inCell)
                i = inCell(ii);
                px = allPoints(i, 1);
                py = allPoints(i, 2);
                onEdge = (abs(px - x0) < 1e-10) || (abs(px - x1) < 1e-10) || ...
                         (abs(py - y0) < 1e-10) || (abs(py - y1) < 1e-10);
                if onEdge
                    edgeIdx(end+1) = i; %#ok<AGROW>
                else
                    interiorIdx(end+1) = i; %#ok<AGROW>
                end
            end

            % ---- Edge points: bilinear interpolation ----
            for ii = 1:length(edgeIdx)
                i = edgeIdx(ii);
                px = allPoints(i, 1);
                py = allPoints(i, 2);

                lx = (px - x0) / (x1 - x0 + eps);
                ly = (py - y0) / (y1 - y0 + eps);
                lx = min(max(lx, 0), 1);
                ly = min(max(ly, 0), 1);

                w = (1-lx)*(1-ly)*z00 + lx*(1-ly)*z10 + (1-lx)*ly*z01 + lx*ly*z11;
                w_sum = sum(w);
                if w_sum > 0
                    M_out(i, :) = w / w_sum;
                end
            end

            % ---- Interior points: LP ----
            if isempty(interiorIdx)
                continue;
            end

            nInt = length(interiorIdx);  % number of interior points

            % All points involved in LP: 4 corners (fixed) + interior (variable)
            % We only optimize over interior points.
            % Variable layout: x = [p_{1,1}, ..., p_{1,nY}, p_{2,1}, ..., p_{nInt,nY}]
            %                       (nInt * nY variables)

            cornerIdx_all = [i00, i10, i01, i11];
            cornerZ_all   = [z00; z10; z01; z11];  % 4 x nY
            cornerCoords  = cornerPoints(cornerIdx_all, :);  % 4 x 2

            intCoords = allPoints(interiorIdx, :);  % nInt x 2
            intLoss   = loss_matrix_selected(interiorIdx, :);  % nInt x nY

            nVar = nInt * nY;

            % -- Objective: minimize expected loss over interior points --
            % cost(k*nY + y) = loss(k, y) / nAll  (average over all points)
            c_obj = zeros(nVar, 1);
            for k = 1:nInt
                c_obj((k-1)*nY+1 : k*nY) = intLoss(k, :)';
            end

            % -- Equality constraints: each interior point sums to 1 --
            Aeq = zeros(nInt, nVar);
            beq = ones(nInt, 1);
            for k = 1:nInt
                Aeq(k, (k-1)*nY+1 : k*nY) = 1;
            end

            % -- Inequality constraints: mDP between all pairs --
            % Pairs to consider:
            %   (a) corner - interior
            %   (b) interior - interior
            % For corners, z is fixed (constant), so we move it to b side.
            %
            % mDP constraint between point i (z_i) and point j (z_j):
            %   z_i(y) <= exp(eps * d(i,j)) * z_j(y)  for all y
            %   z_j(y) <= exp(eps * d(i,j)) * z_i(y)  for all y

            A_ineq = [];
            b_ineq = [];

            % (a) corner (fixed) vs interior (variable)
            for ci = 1:4
                zc = cornerZ_all(ci, :);  % 1 x nY, fixed
                pc = cornerCoords(ci, :);
                for k = 1:nInt
                    pk = intCoords(k, :);
                    d  = norm(pc - pk);
                    a_val = exp(epsilon * d);

                    for y = 1:nY
                        var_ky = (k-1)*nY + y;

                        % z_k(y) <= a_val * zc(y)  =>  z_k(y) <= a_val * zc(y)
                        row1 = zeros(1, nVar);
                        row1(var_ky) = 1;
                        A_ineq(end+1, :) = row1; %#ok<AGROW>
                        b_ineq(end+1)    = a_val * zc(y); %#ok<AGROW>

                        % zc(y) <= a_val * z_k(y)  =>  -z_k(y) <= -zc(y)/a_val
                        row2 = zeros(1, nVar);
                        row2(var_ky) = -a_val;
                        A_ineq(end+1, :) = row2; %#ok<AGROW>
                        b_ineq(end+1)    = -zc(y); %#ok<AGROW>
                    end
                end
            end

            % (b) interior vs interior
            for k1 = 1:nInt
                for k2 = k1+1:nInt
                    pk1 = intCoords(k1, :);
                    pk2 = intCoords(k2, :);
                    d   = norm(pk1 - pk2);
                    a_val = exp(epsilon * d);

                    for y = 1:nY
                        var_k1y = (k1-1)*nY + y;
                        var_k2y = (k2-1)*nY + y;

                        % z_k1(y) <= a_val * z_k2(y)
                        row1 = zeros(1, nVar);
                        row1(var_k1y) =  1;
                        row1(var_k2y) = -a_val;
                        A_ineq(end+1, :) = row1; %#ok<AGROW>
                        b_ineq(end+1)    = 0; %#ok<AGROW>

                        % z_k2(y) <= a_val * z_k1(y)
                        row2 = zeros(1, nVar);
                        row2(var_k2y) =  1;
                        row2(var_k1y) = -a_val;
                        A_ineq(end+1, :) = row2; %#ok<AGROW>
                        b_ineq(end+1)    = 0; %#ok<AGROW>
                    end
                end
            end

            lb = zeros(nVar, 1);
            ub = ones(nVar, 1);

            options = optimoptions('linprog', 'Display', 'off');
            x_sol = linprog(c_obj, A_ineq, b_ineq(:), Aeq, beq, lb, ub, options);

            if isempty(x_sol)
                % LP infeasible: fall back to bilinear interpolation
                for k = 1:nInt
                    i = interiorIdx(k);
                    px = allPoints(i, 1);
                    py = allPoints(i, 2);
                    lx = min(max((px-x0)/(x1-x0+eps), 0), 1);
                    ly = min(max((py-y0)/(y1-y0+eps), 0), 1);
                    w  = (1-lx)*(1-ly)*z00 + lx*(1-ly)*z10 + (1-lx)*ly*z01 + lx*ly*z11;
                    w_sum = sum(w);
                    if w_sum > 0
                        M_out(interiorIdx(k), :) = w / w_sum;
                    end
                end
            else
                for k = 1:nInt
                    z_k = x_sol((k-1)*nY+1 : k*nY)';
                    z_sum = sum(z_k);
                    if z_sum > 0
                        M_out(interiorIdx(k), :) = z_k / z_sum;
                    end
                end
            end

        end  % col_g
    end  % row_g

    if nargin < 6 || isempty(prior)
        loss = sum(sum(M_out .* loss_matrix_selected)) / nAll;
    else
        loss = prior' * sum(M_out .* loss_matrix_selected, 2);
    end
end