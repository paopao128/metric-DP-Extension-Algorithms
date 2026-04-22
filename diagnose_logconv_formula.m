%% Minimal dimension test for logconv_interp formula
addpath('./functions/haversine'); addpath('./functions'); addpath('./functions/benchmarks');
parameters;
city_list = {'rome'}; city_idx = 1; city = city_list{city_idx};
node_file = sprintf('./datasets/%s/nodes.csv', city);
opts = detectImportOptions(node_file); opts = setvartype(opts, 'osmid', 'int64');
df_nodes = readtable(node_file, opts);
col_longitude_orig = table2array(df_nodes(:,'x')); col_latitude_orig = table2array(df_nodes(:,'y'));
max_longitude=max(col_longitude_orig); min_longitude=min(col_longitude_orig); mid_longitude=(max_longitude+min_longitude)/2; LONGITUDE_SIZE=max_longitude-min_longitude;
max_latitude=max(col_latitude_orig); min_latitude=min(col_latitude_orig); mid_latitude=(max_latitude+min_latitude)/2; LATITUDE_SIZE=max_latitude-min_latitude;
lon_range=[mid_longitude-LONGITUDE_SIZE/SCALE, mid_longitude+LONGITUDE_SIZE/SCALE];
lat_range=[mid_latitude-LATITUDE_SIZE/SCALE, mid_latitude+LATITUDE_SIZE/SCALE];
selected_indices=filter_coords_by_range(col_longitude_orig,col_latitude_orig,lon_range,lat_range);
col_longitude=col_longitude_orig(selected_indices,:); col_latitude=col_latitude_orig(selected_indices,:);
[col_longitude, col_latitude]=lonlat_to_xy(col_longitude,col_latitude,mid_longitude,mid_latitude);
NR_REAL_LOC=size(col_longitude,1);
rng(42);
perturbed_indices=randi([1,NR_REAL_LOC],1,NR_PER_LOC);
perturbed_xy=[col_latitude(perturbed_indices',1), col_longitude(perturbed_indices',1)];
loss_matrix=pdist2([col_latitude,col_longitude],perturbed_xy);
loss_matrix_max=min(loss_matrix,[],2);
[adjMatrix,distanceMatrix,neighborMatrix,cornerPoints_ori,~,~,~,corner_weights]=uniform_anchor(col_latitude,col_longitude,loss_matrix_max,cell_size(1,city_idx));
close all;
EPSILON = 0.5; epsilon_idx_1 = 2;
epsilon_1 = EPSILON*epsilon_idx_1/NR_EPSILON_INTERVAL;
epsilon_2 = EPSILON*sqrt(1-(epsilon_idx_1/NR_EPSILON_INTERVAL)^2);
c_approx = corner_weights' * loss_matrix;
[z_anchor, ~] = perturbation_cal_apo(c_approx, corner_weights, distanceMatrix, neighborMatrix, epsilon_1/2, epsilon_2/2);
cornerPoints = cornerPoints_ori; allPoints = cornerPoints_ori;
cornerPoints = allPoints; interpolation_extension_k;
W_test = full(W);
distMat = pdist2(allPoints, perturbed_xy);

fprintf('===== SIZES =====\n');
fprintf('z_anchor:  %s\n', mat2str(size(z_anchor)));
fprintf('W (sparse): %s\n', mat2str(size(W)));
fprintf('W_test (full): %s\n', mat2str(size(W_test)));
fprintf('distMat:   %s\n', mat2str(size(distMat)));

i_test=1; k_test=1;
a1 = z_anchor(:, k_test);         % should be J×1
a2 = z_anchor(:, k_test)';        % should be 1×J
b1 = W_test(i_test, :)';          % should be J×1
b2 = W_test(i_test, :);           % should be 1×J

fprintf('\n--- Vector shapes ---\n');
fprintf('z_anchor(:,k):   %s\n', mat2str(size(a1)));
fprintf('z_anchor(:,k)'': %s\n', mat2str(size(a2)));
fprintf('W(i,:)'':        %s\n', mat2str(size(b1)));
fprintf('W(i,:):          %s\n', mat2str(size(b2)));

fprintf('\n--- Power operation shapes ---\n');
t1 = a1 .^ b1;  fprintf('(J×1).^(J×1) = %s\n', mat2str(size(t1)));
t2 = a2 .^ b2;  fprintf('(1×J).^(1×J) = %s\n', mat2str(size(t2)));
try
    t3 = a2 .^ b1;  fprintf('(1×J).^(J×1) = %s  ← broadcasts!\n', mat2str(size(t3)));
catch e
    fprintf('(1×J).^(J×1) ERROR: %s\n', e.message);
end
try
    t4 = a1 .^ b2;  fprintf('(J×1).^(1×J) = %s  ← broadcasts!\n', mat2str(size(t4)));
catch e
    fprintf('(J×1).^(1×J) ERROR: %s\n', e.message);
end

fprintf('\n--- prod shapes ---\n');
fprintf('prod(J×1) = %s, val=%.6f\n', mat2str(size(prod(t1))), prod(t1));
fprintf('prod(1×J) = %s, val=%.6f\n', mat2str(size(prod(t2))), prod(t2));

fprintf('\n--- Current logconv_interp code: prod(a2 .^ b1) ---\n');
buggy = a2 .^ b1;  % (1×J).^(J×1) → J×J
fprintf('size(a2 .^ b1) = %s\n', mat2str(size(buggy)));
pb = prod(buggy);   % prod(J×J) → 1×J
fprintf('size(prod(J×J)) = %s\n', mat2str(size(pb)));
fprintf('Is prod(J×J)(1) == z_anchor(1,k)? %.6f vs %.6f\n', pb(1), z_anchor(1,k_test));
fprintf('All elements equal to z_anchor(:,k)''? max_diff = %.2e\n', max(abs(pb - a2)));

fprintf('\n===== logconv_interp ACTUAL OUTPUT =====\n');
[x_curr, loss_curr] = logconv_interp(z_anchor, W_test, distMat);
fprintf('size(x_curr) = %s\n', mat2str(size(x_curr)));
fprintf('loss_curr = %.4f\n', loss_curr);
row_diffs = max(abs(x_curr - x_curr(1,:)), [], 2);
fprintf('Max row-to-row diff in x_curr: %.4f\n', max(row_diffs));

fprintf('\n===== SYNTHETIC SMALL EXAMPLE =====\n');
% Tiny example to understand behavior
J2=3; nFine2=4; K2=2;
z_small = rand(J2, K2);  % J×K
W_small = rand(nFine2, J2);  W_small = W_small ./ sum(W_small,2);  % normalize

% Method A: current (buggy?) code
clear x_A;
for k=1:K2
    for i=1:nFine2
        x_A(i,k) = prod(z_small(:,k)' .^ W_small(i,:)');
    end
end
% Method B: correct code
clear x_B;
for k=1:K2
    for i=1:nFine2
        x_B(i,k) = prod(z_small(:,k) .^ W_small(i,:)');
    end
end
fprintf('x_A size: %s,  x_B size: %s\n', mat2str(size(x_A)), mat2str(size(x_B)));
fprintf('x_A:\n'); disp(x_A);
fprintf('x_B:\n'); disp(x_B);
fprintf('Are they equal? max_diff = %.2e\n', max(abs(x_A(:)-x_B(:))));

% EM loss for comparison
P_em = exp(-EPSILON * distMat / 2); P_em = P_em ./ sum(P_em,2);
loss_em = sum(sum(P_em .* distMat)) / size(distMat,1);
fprintf('\nEM loss = %.4f,  logconv loss = %.4f\n', loss_em, loss_curr);
