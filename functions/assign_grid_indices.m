function idx_grid = assign_grid_indices(lon, lat, min_lon, min_lat, size_x, size_y, num_cells_x, num_cells_y)
    grid_x = floor((lon - min_lon) / size_x) + 1;
    grid_y = floor((lat - min_lat) / size_y) + 1;

    % Clamp indices
    grid_x = max(min(grid_x, num_cells_x), 1);
    grid_y = max(min(grid_y, num_cells_y), 1);

    idx_grid = sub2ind([num_cells_y, num_cells_x], grid_y, grid_x);  % row-major order
end