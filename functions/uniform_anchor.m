function [adjMatrix, distanceMatrix, neighborMatrix, cornerPoints, squares, lambda_x, lambda_y, corner_weights] = uniform_anchor(x, y, cost, cell_size)
    % Vectorized uniform grid partition for improved performance

    % Bounding box
    minX = min(x);  maxX = max(x);
    minY = min(y);  maxY = max(y);

    % Expand the grid to fully cover the range, including max edge
    cell_size = cell_size * 1.0; 
    x_edges = minX:cell_size:(maxX + cell_size);
    y_edges = minY:cell_size:(maxY + cell_size);
    Nx = numel(x_edges);
    Ny = numel(y_edges);

    nPts = numel(x);
    squares(nPts,1) = struct('minX',[],'maxX',[],'minY',[],'maxY',[],'points',[]);
    lambda_x = zeros(nPts,1);
    lambda_y = zeros(nPts,1);

    % Bin indices for each point
    cx_idx = discretize(x, x_edges);
    cy_idx = discretize(y, y_edges);

    % Clamp NaNs (should be rare after expanding edges, but still safe)
    cx_idx(isnan(cx_idx)) = Nx - 1;
    cy_idx(isnan(cy_idx)) = Ny - 1;

    % Ensure no out-of-bound indexing
    cx_idx = min(cx_idx, Nx - 1);
    cy_idx = min(cy_idx, Ny - 1);

    % Cell boundaries per point
    x0 = x_edges(cx_idx);
    x1 = x_edges(cx_idx + 1);
    y0 = y_edges(cy_idx);
    y1 = y_edges(cy_idx + 1);

    % Compute barycentric coordinates
    dx = x1 - x0; dy = y1 - y0;
    lambda_x = (x' - x0) ./ max(dx, eps);
    lambda_y = (y' - y0) ./ max(dy, eps);
    lambda_x = min(max(lambda_x, 0), 1);
    lambda_y = min(max(lambda_y, 0), 1);

    for i = 1:nPts
        squares(i).minX = x0(i);
        squares(i).maxX = x1(i);
        squares(i).minY = y0(i);
        squares(i).maxY = y1(i);
        squares(i).points = [x(i), y(i), cost(i)];
    end

    % Generate corner points
    [Xc, Yc] = meshgrid(x_edges, y_edges);
    cornerPoints = [Xc(:), Yc(:)];
    numCorners = size(cornerPoints, 1);

    % Corner key mapping
    cornerMap = containers.Map('KeyType','char','ValueType','double');
    for k = 1:numCorners
        key = sprintf('%.6f,%.6f', cornerPoints(k,1), cornerPoints(k,2));
        cornerMap(key) = k;
    end

    % Compute corner weights
    corner_weights = sparse(nPts, numCorners);
    for i = 1:nPts
        c = [x0(i), y0(i);
             x1(i), y0(i);
             x0(i), y1(i);
             x1(i), y1(i)];
        w = [(1 - lambda_x(i)) * (1 - lambda_y(i));
             lambda_x(i)     * (1 - lambda_y(i));
             (1 - lambda_x(i)) * lambda_y(i);
             lambda_x(i)     * lambda_y(i)];
        w = max(w, 0);
        w_sum = sum(w);
        if w_sum > 0
            w = w / w_sum;
        end
        for j = 1:4
            key = sprintf('%.6f,%.6f', c(j,1), c(j,2));
            idx = cornerMap(key);
            corner_weights(i, idx) = w(j);
        end
    end

    % Adjacency matrices
    adjMatrix = sparse(numCorners, numCorners);
    neighborMatrix = sparse(numCorners, numCorners);
    for row = 1:Ny
        for col = 1:Nx-1
            a = row + (col-1)*Ny;
            b = row + col*Ny;
            adjMatrix(a,b) = 1; adjMatrix(b,a) = 1;
            neighborMatrix(a,b) = 1; neighborMatrix(b,a) = 1;
        end
    end
    for col = 1:Nx
        for row = 1:Ny-1
            a = row + (col-1)*Ny;
            b = (row+1) + (col-1)*Ny;
            adjMatrix(a,b) = 1; adjMatrix(b,a) = 1;
            neighborMatrix(a,b) = 2; neighborMatrix(b,a) = 2;
        end
    end

    % Distance matrix
    [r, c] = find(adjMatrix);
    distanceMatrix = sparse(numCorners, numCorners);
    for k = 1:length(r)
        distanceMatrix(r(k), c(k)) = norm(cornerPoints(r(k),:) - cornerPoints(c(k),:));
    end

    % Visualizations
    visualize_grid_cells(squares, x, y);
    visualize_adjacency_graph(adjMatrix, cornerPoints);
end

function visualize_grid_cells(squares, x, y)
    figure('Name', 'Uniform Grid Partitions'); hold on;
    scatter(x, y, 20, 'filled');
    title('Uniform Grid Squares');
    for i = 1:numel(squares)
        plot([squares(i).minX, squares(i).maxX, squares(i).maxX, squares(i).minX, squares(i).minX], ...
             [squares(i).minY, squares(i).minY, squares(i).maxY, squares(i).maxY, squares(i).minY], 'k');
    end
    hold off;
end

function visualize_adjacency_graph(adjMatrix, cornerPoints)
    figure('Name', 'Adjacency Graph'); hold on;
    [i, j] = find(adjMatrix);
    for k = 1:length(i)
        x_vals = [cornerPoints(i(k),1), cornerPoints(j(k),1)];
        y_vals = [cornerPoints(i(k),2), cornerPoints(j(k),2)];
        plot(x_vals, y_vals, 'b-');
    end
    scatter(cornerPoints(:,1), cornerPoints(:,2), 15, 'filled');
    title('Adjacency Graph of Uniform Grid');
    xlabel('X Coordinate'); ylabel('Y Coordinate');
    axis equal; grid on; hold off;
end
