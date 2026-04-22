% 
% cornerPoints: 132x2
% k k^2
k=2;

xUnique = sort(unique(cornerPoints(:,1)));
yUnique = sort(unique(cornerPoints(:,2)));
n = length(yUnique);
m = length(xUnique);

origIdx = zeros(n, m);
for t = 1:size(cornerPoints, 1)
    col = find(xUnique == cornerPoints(t,1));
    row = find(yUnique == cornerPoints(t,2));
    origIdx(row, col) = t;
end

xAll = [];
for i = 1:m-1
    xAll = [xAll, linspace(xUnique(i), xUnique(i+1), k+1)];
end
xAll = unique(xAll)';

yAll = [];
for i = 1:n-1
    yAll = [yAll, linspace(yUnique(i), yUnique(i+1), k+1)];
end
yAll = unique(yAll)';

nAll = length(yAll);
mAll = length(xAll);
totalNew = nAll * mAll;
nOrig = size(cornerPoints, 1);

W = sparse(totalNew, nOrig);

for ii = 1:nAll
    for jj = 1:mAll
        px = xAll(jj);
        py = yAll(ii);
        ptIdx = (jj - 1) * nAll + ii;  % column-major: matches meshgrid(:) ordering of allPoints

        cx = find(xUnique <= px, 1, 'last');
        if cx >= m, cx = m - 1; end
        cy = find(yUnique <= py, 1, 'last');
        if cy >= n, cy = n - 1; end

        x0 = xUnique(cx);   x1 = xUnique(cx+1);
        y0 = yUnique(cy);   y1 = yUnique(cy+1);

        if x1 ~= x0
            lx = (px - x0) / (x1 - x0);
        else
            lx = 0;
        end
        if y1 ~= y0
            ly = (py - y0) / (y1 - y0);
        else
            ly = 0;
        end

        idx_00 = origIdx(cy,   cx);
        idx_10 = origIdx(cy,   cx+1);
        idx_01 = origIdx(cy+1, cx);
        idx_11 = origIdx(cy+1, cx+1);

        W(ptIdx, idx_00) = (1 - lx) * (1 - ly);
        W(ptIdx, idx_10) = lx       * (1 - ly);
        W(ptIdx, idx_01) = (1 - lx) * ly;
        W(ptIdx, idx_11) = lx       * ly;
    end
end

[XX, YY] = meshgrid(xAll, yAll);
allPoints = [XX(:), YY(:)];

% disp(['加密倍数 k = ', num2str(k)]);
% disp(['原始网格: ', num2str(n), ' x ', num2str(m), ' = ', num2str(nOrig), ' 点']);
% disp(['加密网格: ', num2str(nAll), ' x ', num2str(mAll), ' = ', num2str(totalNew), ' 点']);
% disp(['W 矩阵大小: ', num2str(size(W,1)), ' x ', num2str(size(W,2))]);
% disp(['权重行和范围: ', num2str(min(full(sum(W,2)))), ' ~ ', num2str(max(full(sum(W,2))))]);