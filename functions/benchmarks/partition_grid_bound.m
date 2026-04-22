function [grid_utility_loss, grid_distances, grid_prior, neighbor_pairs, distances_to_perturbed] = ...
    partition_grid_bound(longitude, latitude, utility_loss_matrix, prior, N, M, perturbed_lat, perturbed_lon, l_p)
%PARTITION_GRID_BOUND  Partition points into an N-by-M grid and compute:
%  - grid_utility_loss(i,k): min utility loss in cell i for perturbed loc k
%  - grid_distances(i,j): max corner‐to‐corner distance between cell i and j
%  - neighbor_pairs: 4‐connected grid adjacency
%  - distances_to_perturbed(i,j): distance from centroid of cell i to perturbed j

    %% 1) basic checks & setup
    num_locations = numel(longitude);
    if size(utility_loss_matrix,1) ~= num_locations
        error('utility_loss_matrix must have one row per input location');
    end

    % 2) build grid edges
    lon_edges = linspace(min(longitude), max(longitude), M+1);
    lat_edges = linspace(min(latitude),  max(latitude),  N+1);

    % 3) assign each input point to a cell index 1..N*M (or 0 if outside)
    location_cell_indices = zeros(num_locations,1);
    for i = 1:num_locations
        % find the bin, or 0 if none
        li = find(longitude(i) < lon_edges,1,'first') - 1;
        if isempty(li), li = 0; end
        lj = find(latitude(i)  < lat_edges,1,'first') - 1;
        if isempty(lj), lj = 0; end

        if li>=1 && li<=M && lj>=1 && lj<=N
            location_cell_indices(i) = sub2ind([N,M], lj, li);
        else
            location_cell_indices(i) = 0;
        end
    end

    %% 4) grid_utility_loss = minimum over each cell
    num_cells = N*M;
    num_perturbed = size(utility_loss_matrix,2);
    grid_utility_loss = zeros(num_cells, num_perturbed);
    grid_prior = zeros(1,N*M); 
    for cell_id = 1:num_cells
        idx = find(location_cell_indices == cell_id);
        grid_prior(cell_id) = sum(prior(idx)); 

        if ~isempty(idx)
            % take min across those rows
            grid_utility_loss(cell_id,:) = min( utility_loss_matrix(idx,:), [], 1 ); % *size(idx, 1);
            % for k = 1:1:num_perturbed
            %     if grid_utility_loss(cell_id,k) == 0
            %         grid_utility_loss(cell_id,:) = max(grid_utility_loss(cell_id,:), 2000*size(idx, 1));
            %     end
            % end
        end
    end

    %% 5) neighbor_pairs (4‐connected)
    neighbor_pairs = [];
    for r = 1:N
      for c = 1:M
        id = sub2ind([N,M],r,c);
        if c<M, neighbor_pairs(end+1,:) = [id, sub2ind([N,M],r,c+1)]; end
        if r<N, neighbor_pairs(end+1,:) = [id, sub2ind([N,M],r+1,c)]; end
      end
    end

    %% 6) cell centroids (for distances_to_perturbed)
    lon_centers = (lon_edges(1:end-1)+lon_edges(2:end))/2;
    lat_centers = (lat_edges(1:end-1)+lat_edges(2:end))/2;
    [LonG, LatG] = meshgrid(lon_centers, lat_centers);
    centroids = [LatG(:), LonG(:)];  % [lat lon]

    distances_to_perturbed = zeros(num_cells, num_perturbed);
    for i = 1:num_cells
      for j = 1:num_perturbed
        distances_to_perturbed(i,j) = norm(centroids(i,:) - [perturbed_lat(j), perturbed_lon(j)], l_p);
      end
    end

    %% 7) precompute each cell's 4 corners
    % corners{i} is a 4x2 array [lat lon] of the cell's corners
    corners = cell(num_cells,1);
    for r = 1:N
      for c = 1:M
        id = sub2ind([N,M], r, c);
        lon0 = lon_edges(c);   lon1 = lon_edges(c+1);
        lat0 = lat_edges(r);   lat1 = lat_edges(r+1);
        % 4 corners:
        corners{id} = [ lat0 lon0;
                        lat0 lon1;
                        lat1 lon0;
                        lat1 lon1 ];
      end
    end

    %% 8) grid_distances(i,j) = max distance between any corner of cell i and any corner of cell j
    grid_distances = zeros(num_cells);
    for i = 1:num_cells
      for j = i+1:num_cells
        % all pairwise corner distances
        D = pdist2(corners{i}, corners{j});  
        dmax = max(D(:));
        grid_distances(i,j) = dmax;
        grid_distances(j,i) = dmax;
      end
    end

end
