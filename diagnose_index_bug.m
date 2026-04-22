%% Verify the W vs allPoints index mismatch in interpolation_extension_k
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
[~,distanceMatrix,neighborMatrix,cornerPoints_ori,~,~,~,corner_weights]=uniform_anchor(col_latitude,col_longitude,loss_matrix_max,cell_size(1,city_idx));
close all;

fprintf('===== Grid dimensions =====\n');
xU = sort(unique(cornerPoints_ori(:,1))); yU = sort(unique(cornerPoints_ori(:,2)));
Nx = length(xU); Ny = length(yU);
fprintf('Original cornerPoints: %d = %d(Ny) x %d(Nx)\n', size(cornerPoints_ori,1), Ny, Nx);

%% Run one round of interpolation_extension_k (CURRENT buggy version)
cornerPoints = cornerPoints_ori;
allPoints = cornerPoints_ori;
cornerPoints = allPoints;
interpolation_extension_k;  % produces W, allPoints with current code
W_buggy = W;
allPoints_result = allPoints;

fprintf('After 1 round: allPoints = %d x 2, W = %d x %d\n', size(allPoints,1), size(W,1), size(W,2));
xAllU = sort(unique(allPoints(:,1))); yAllU = sort(unique(allPoints(:,2)));
fprintf('nAll(Ny_fine)=%d, mAll(Nx_fine)=%d\n', length(yAllU), length(xAllU));
nAll_val = length(yAllU); mAll_val = length(xAllU);

%% KEY TEST: for each row i of W, the weighted sum of cornerPoints should = allPoints(i,:)
% If W(i,:) are bilinear weights for the point at allPoints(i,:), then:
%   W_buggy(i,:) * cornerPoints_ori ≈ allPoints_result(i,:)
reconstructed = full(W_buggy) * cornerPoints_ori;  % (totalNew x 2)
err = reconstructed - allPoints_result;
max_err = max(abs(err(:)));
mean_err = mean(abs(err(:)));
fprintf('\n--- With CURRENT (buggy) ptIdx = (ii-1)*mAll + jj ---\n');
fprintf('Max reconstruction error:  %.6f\n', max_err);
fprintf('Mean reconstruction error: %.6f\n', mean_err);
n_wrong = sum(abs(err(:,1)) > 1e-6 | abs(err(:,2)) > 1e-6);
fprintf('Points with error > 1e-6:  %d / %d (%.1f%%)\n', n_wrong, size(allPoints_result,1), 100*n_wrong/size(allPoints_result,1));

%% Now build W_fixed with column-major ptIdx = (jj-1)*nAll + ii
cornerPoints = cornerPoints_ori;  % reset
xUnique = sort(unique(cornerPoints(:,1)));
yUnique = sort(unique(cornerPoints(:,2)));
n = length(yUnique); m = length(xUnique);
origIdx = zeros(n, m);
for t = 1:size(cornerPoints,1)
    col = find(xUnique == cornerPoints(t,1));
    row = find(yUnique == cornerPoints(t,2));
    origIdx(row, col) = t;
end
k_val = 2;
xAll2=[]; for i=1:m-1, xAll2=[xAll2, linspace(xUnique(i),xUnique(i+1),k_val+1)]; end
xAll2=unique(xAll2)';
yAll2=[]; for i=1:n-1, yAll2=[yAll2, linspace(yUnique(i),yUnique(i+1),k_val+1)]; end
yAll2=unique(yAll2)';
nAll2=length(yAll2); mAll2=length(xAll2); totalNew2=nAll2*mAll2;
nOrig=size(cornerPoints,1);

W_fixed = sparse(totalNew2, nOrig);
for ii=1:nAll2
    for jj=1:mAll2
        px=xAll2(jj); py=yAll2(ii);
        ptIdx_fixed = (jj-1)*nAll2 + ii;  % column-major: matches meshgrid(:)
        cx=find(xUnique<=px,1,'last'); if cx>=m, cx=m-1; end
        cy=find(yUnique<=py,1,'last'); if cy>=n, cy=n-1; end
        x0=xUnique(cx); x1=xUnique(cx+1); y0=yUnique(cy); y1=yUnique(cy+1);
        lx=0; ly=0;
        if x1~=x0, lx=(px-x0)/(x1-x0); end
        if y1~=y0, ly=(py-y0)/(y1-y0); end
        idx_00=origIdx(cy,cx); idx_10=origIdx(cy,cx+1);
        idx_01=origIdx(cy+1,cx); idx_11=origIdx(cy+1,cx+1);
        W_fixed(ptIdx_fixed,idx_00)=(1-lx)*(1-ly);
        W_fixed(ptIdx_fixed,idx_10)=lx*(1-ly);
        W_fixed(ptIdx_fixed,idx_01)=(1-lx)*ly;
        W_fixed(ptIdx_fixed,idx_11)=lx*ly;
    end
end
[XX2,YY2]=meshgrid(xAll2,yAll2);
allPoints_fixed=[XX2(:),YY2(:)];

reconstructed_fixed = full(W_fixed) * cornerPoints_ori;
err_fixed = reconstructed_fixed - allPoints_fixed;
max_err_fixed = max(abs(err_fixed(:)));
mean_err_fixed = mean(abs(err_fixed(:)));
fprintf('\n--- With FIXED ptIdx = (jj-1)*nAll + ii ---\n');
fprintf('Max reconstruction error:  %.6f\n', max_err_fixed);
fprintf('Mean reconstruction error: %.6f\n', mean_err_fixed);
n_wrong_fixed = sum(abs(err_fixed(:,1))>1e-6 | abs(err_fixed(:,2))>1e-6);
fprintf('Points with error > 1e-6:  %d / %d (%.1f%%)\n', n_wrong_fixed, size(allPoints_fixed,1), 100*n_wrong_fixed/size(allPoints_fixed,1));

%% Show impact on loss: use same LP solution and compare losses
c_approx = pdist2(cornerPoints_ori, perturbed_xy);
EPSILON=0.5; epsilon_idx_1=2;
epsilon_1=EPSILON*epsilon_idx_1/NR_EPSILON_INTERVAL;
epsilon_2=EPSILON*sqrt(1-(epsilon_idx_1/NR_EPSILON_INTERVAL)^2);
[z_anchor, fval] = perturbation_cal_apo(c_approx, corner_weights, distanceMatrix, neighborMatrix, epsilon_1/2, epsilon_2/2);
fprintf('\n===== Loss comparison =====\n');
fprintf('loss_aipo_instance_approx = %.4f\n', fval/NR_REAL_LOC);

[~, loss_buggy]  = logconv_interp(z_anchor, W_buggy,  pdist2(allPoints_result, perturbed_xy));
[~, loss_fixed]  = logconv_interp(z_anchor, W_fixed,  pdist2(allPoints_fixed, perturbed_xy));
fprintf('loss with BUGGY  W (1 round): %.4f\n', loss_buggy);
fprintf('loss with FIXED  W (1 round): %.4f\n', loss_fixed);
