function p_fine = convert_grid_to_fine_perturbation( ...
    p_grid, ...
    col_longitude, col_latitude, ...
    num_cells_x, num_cells_y)

    %% === Step 1: Define Grid Boundaries ===
    min_lon = min(col_longitude);
    max_lon = max(col_longitude);
    min_lat = min(col_latitude);
    max_lat = max(col_latitude);

    size_x = (max_lon - min_lon) / num_cells_x;
    size_y = (max_lat - min_lat) / num_cells_y;

    %% === Step 2: Map Fine Locations to Grid Cells ===
    idx_input_grid = assign_grid_indices(col_longitude, col_latitude, ...
                                         min_lon, min_lat, size_x, size_y, ...
                                         num_cells_x, num_cells_y);

    n = size(p_grid, 1);  % # input grid cells
    m = size(p_grid, 2);  % # output grid cells
    N = length(col_longitude);  % # fine-grained input points

    %% === Step 3: Expand to Fine-grained Perturbation Matrix ===
    p_fine = zeros(N, m);
    for i = 1:N
        p_fine(i, :) = p_grid(idx_input_grid(i), :);
    end
end
