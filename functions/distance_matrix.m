function D = distance_matrix(latitudes, longitudes, l_p)
% distance_matrix_2norm - Compute pairwise Euclidean distance matrix
%
% Inputs:
%    latitudes  - Column vector of latitudes (n x 1)
%    longitudes - Column vector of longitudes (n x 1)
%
% Output:
%    D - (n x n) symmetric matrix of Euclidean distances

n = length(latitudes);
D = zeros(n);

coords = [latitudes(:), longitudes(:)]; % n x 2 matrix

% Compute pairwise distances
for i = 1:n
    for j = i+1:n
        diff = coords(i,:) - coords(j,:);
        d = norm(diff, l_p);  % Euclidean (L2) norm
        D(i,j) = d;
        D(j,i) = d; % Symmetric
    end
end

end