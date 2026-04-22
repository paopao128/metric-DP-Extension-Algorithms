function [x, y] = lonlat_to_xy(lon, lat, lon0, lat0)
% Converts longitude/latitude to local x/y coordinates (in km)
% using an equirectangular projection (suitable for small regions).
%
% Inputs:
%   lon, lat - vectors or matrices of longitude and latitude (in degrees)
%   lon0, lat0 - reference center point (in degrees)
%
% Outputs:
%   x, y - Cartesian coordinates (in km), relative to center

    % Earth radius in kilometers
    R = 6371;

    % Convert degrees to radians
    lon = deg2rad(lon);
    lat = deg2rad(lat);
    lon0 = deg2rad(lon0);
    lat0 = deg2rad(lat0);

    % Project to local coordinates
    x = R * (lon - lon0) .* cos((lat + lat0)/2);
    y = R * (lat - lat0);
end
