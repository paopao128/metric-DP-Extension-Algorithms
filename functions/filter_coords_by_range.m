function indices = filter_coords_by_range(lon_vec, lat_vec, lon_range, lat_range)
% FILTER_COORDS_BY_RANGE_IDX Return indices of coordinates within given longitude and latitude ranges
%
% Inputs:
%   lon_vec   - vector of longitudes
%   lat_vec   - vector of latitudes (same length as lon_vec)
%   lon_range - 2-element vector [min_lon, max_lon]
%   lat_range - 2-element vector [min_lat, max_lat]
%
% Output:
%   indices - indices of elements within the specified lat/lon ranges

    % Input validation
    if length(lon_vec) ~= length(lat_vec)
        error('Longitude and latitude vectors must be the same length.');
    end
    if length(lon_range) ~= 2 || length(lat_range) ~= 2
        error('Longitude and latitude ranges must be 2-element vectors.');
    end

    % Logical mask for filtering
    mask = lon_vec >= lon_range(1) & lon_vec <= lon_range(2) & ...
           lat_vec >= lat_range(1) & lat_vec <= lat_range(2);

    % Return indices where condition holds
    indices = find(mask);
end
