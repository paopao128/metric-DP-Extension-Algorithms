function [refinedPoints, refinedZ] = refineGridLogLinear(cornerPoints, z_anchor_instance)
% refineGridLogLinear
% Refine a regular 2D grid by one level:
% 1) Retain original vertices
% 2) Insert midpoints on each edge
% 3) Insert center points in each cell
% Probability distributions are interpolated using log-linear interpolation
%
% Inputs:
%   cornerPoints      : N x 2, vertex coordinates
%   z_anchor_instance : N x K, probability distribution at each vertex
%
% Outputs:
%   refinedPoints : N_new x 2, coordinates of all refined points
%   refinedZ      : N_new x K, probability distributions at all refined points

    %-----------------------------
    % Parameter validation
    %-----------------------------
    [N, dim] = size(cornerPoints);
    if dim ~= 2
        error('cornerPoints must be an N x 2 matrix');
    end

    if size(z_anchor_instance,1) ~= N
        error('Number of rows in z_anchor_instance must match cornerPoints');
    end

    K = size(z_anchor_instance,2);

    % Prevent log(0)
    eps_val = 1e-300;
    Z = max(z_anchor_instance, eps_val);

    %-----------------------------
    % Recover regular grid structure
    %-----------------------------
    xVals = unique(cornerPoints(:,1));
    yVals = unique(cornerPoints(:,2));

    nx = length(xVals);
    ny = length(yVals);

    if nx * ny ~= N
        error(['cornerPoints does not appear to be a complete regular rectangular grid: ' ...
               'unique(x)*unique(y) ~= number of points']);
    end

    % Build a mapping from (ix,iy) -> original point index
    % Assume every (x,y) combination exists and is unique
    indexMap = zeros(ny, nx);
    % Rows correspond to y, columns correspond to x

    tol = 1e-12;
    for i = 1:N
        x = cornerPoints(i,1);
        y = cornerPoints(i,2);

        ix = find(abs(xVals - x) < tol, 1);
        iy = find(abs(yVals - y) < tol, 1);

        if isempty(ix) || isempty(iy)
            error('Failed to construct indexMap');
        end

        indexMap(iy, ix) = i;
    end

    if any(indexMap(:) == 0)
        error('Missing grid points detected; cannot form a complete regular grid');
    end

    %-----------------------------
    % Refined grid coordinate axes
    % Original grid nx x ny --> refined (2*nx-1) x (2*ny-1)
    %-----------------------------
    xFine = zeros(1, 2*nx-1);
    yFine = zeros(1, 2*ny-1);

    xFine(1:2:end) = xVals;
    yFine(1:2:end) = yVals;

    for ix = 1:nx-1
        xFine(2*ix) = 0.5*(xVals(ix) + xVals(ix+1));
    end

    for iy = 1:ny-1
        yFine(2*iy) = 0.5*(yVals(iy) + yVals(iy+1));
    end

    nxFine = length(xFine);
    nyFine = length(yFine);

    refinedPoints = zeros(nxFine*nyFine, 2);
    refinedZ = zeros(nxFine*nyFine, K);

    %-----------------------------
    % Populate the refined grid
    % Classification:
    % 1) odd-odd   : original vertices
    % 2) odd-even  : vertical edge midpoints
    % 3) even-odd  : horizontal edge midpoints
    % 4) even-even : cell center points
    %
    % Note: index order here is (iyFine, ixFine)
    %-----------------------------
    cnt = 0;
    for iyFine = 1:nyFine
        for ixFine = 1:nxFine
            cnt = cnt + 1;

            x = xFine(ixFine);
            y = yFine(iyFine);

            refinedPoints(cnt,:) = [x, y];

            %---------------------------------
            % Case 1: original vertex
            %---------------------------------
            if mod(ixFine,2)==1 && mod(iyFine,2)==1
                ix = (ixFine + 1)/2;
                iy = (iyFine + 1)/2;
                idx = indexMap(iy, ix);
                refinedZ(cnt,:) = Z(idx,:);

            %---------------------------------
            % Case 2: horizontal edge midpoint
            % Located between two left/right vertices
            %---------------------------------
            elseif mod(ixFine,2)==0 && mod(iyFine,2)==1
                ixL = ixFine/2;
                ixR = ixL + 1;
                iy  = (iyFine + 1)/2;

                idx1 = indexMap(iy, ixL);
                idx2 = indexMap(iy, ixR);

                p1 = Z(idx1,:);
                p2 = Z(idx2,:);

                % ln(p)=0.5ln(p1)+0.5ln(p2)
                p = exp(0.5*log(p1) + 0.5*log(p2));

                % Normalize to a probability distribution
                p = p / sum(p);
                refinedZ(cnt,:) = p;

            %---------------------------------
            % Case 3: vertical edge midpoint
            % Located between two top/bottom vertices
            %---------------------------------
            elseif mod(ixFine,2)==1 && mod(iyFine,2)==0
                ix = (ixFine + 1)/2;
                iyB = iyFine/2;
                iyT = iyB + 1;

                idx1 = indexMap(iyB, ix);
                idx2 = indexMap(iyT, ix);

                p1 = Z(idx1,:);
                p2 = Z(idx2,:);

                p = exp(0.5*log(p1) + 0.5*log(p2));

                p = p / sum(p);
                refinedZ(cnt,:) = p;

            %---------------------------------
            % Case 4: cell center point
            % Interpolated from the four corner vertices
            %---------------------------------
            else
                ixL = ixFine/2;
                ixR = ixL + 1;
                iyB = iyFine/2;
                iyT = iyB + 1;

                idx1 = indexMap(iyB, ixL); % bottom-left
                idx2 = indexMap(iyB, ixR); % bottom-right
                idx3 = indexMap(iyT, ixL); % top-left
                idx4 = indexMap(iyT, ixR); % top-right

                p1 = Z(idx1,:);
                p2 = Z(idx2,:);
                p3 = Z(idx3,:);
                p4 = Z(idx4,:);

                % ln(p)=0.25*(lnp1+lnp2+lnp3+lnp4)
                p = exp(0.25*(log(p1) + log(p2) + log(p3) + log(p4)));

                p = p / sum(p);
                refinedZ(cnt,:) = p;
            end
        end
    end
end