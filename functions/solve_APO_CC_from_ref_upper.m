function sol = solve_APO_CC_from_ref_upper(xi, epsilon1, epsilon2, distanceMatrix, neighborMatrix, membershipCorners, deltaMin, deltaMax)
% Minimize UPPER bound (w2_tilde) with ε-tight min/max, global t, shared anchors.
% CHANGE: per-cell normalization uses LOWER bound: sum_k w1_tilde(m,k) = 1.

    if nargin < 7 || isempty(deltaMin), deltaMin = 1e-8; end
    if nargin < 8 || isempty(deltaMax), deltaMax = 1e-8; end

    % ---------- sizes & groups ----------
    J = size(distanceMatrix,1);
    [Jm, M] = size(membershipCorners);
    assert(Jm==J, 'membershipCorners must be J x M.');
    hatX_groups = cell(1,M);
    for m = 1:M, hatX_groups{m} = find(membershipCorners(:,m) ~= 0); end
    keep = ~cellfun(@isempty, hatX_groups);
    hatX_groups = hatX_groups(keep); membershipCorners = membershipCorners(:,keep);
    M = numel(hatX_groups);

    if isvector(xi), xi = repmat(xi(:).', M, 1); else, assert(size(xi,1)==M); end
    K = size(xi,2);

    % ---------- variables ----------
    nm = cellfun(@numel, hatX_groups);
    idx = struct; offset = 0;
    for m = 1:M
        idx.z{m}  = offset + (1:(nm(m)*K)); offset = offset + nm(m)*K;
        idx.w1{m} = offset + (1:K);         offset = offset + K;       % lower bound (tilde ω')
        idx.w2{m} = offset + (1:K);         offset = offset + K;       % upper bound (tilde ω'')
    end
    idx.t = offset + 1; offset = offset + 1; nvar = offset;            % global t

    % ---------- objective: minimize upper bound ----------
    f = sparse(nvar,1);
    for m=1:M, f(idx.w2{m}) = xi(m,:).'; end

    lb = zeros(nvar,1); ub = inf(nvar,1);

    % ---------- equalities ----------
    % (1) sum_k z(i,k) = t  (per anchor i in each cell)
    rowsE = sum(nm) + M;                             % +M for per-cell normalization
    Aeq = spalloc(rowsE, nvar, rowsE*(K+1)); beq = zeros(rowsE,1); r = 0;
    for m=1:M
        for ii=1:nm(m)
            r = r+1; zrow = idx.z{m}((ii-1)*K + (1:K));
            Aeq(r, zrow) = 1; Aeq(r, idx.t) = -1;    % global t
        end
    end
    % (2) CHANGED: sum_k w1_tilde(m,k) = 1  (normalize LOWER bound per cell)
    for m=1:M, r=r+1; Aeq(r, idx.w1{m}) = 1; beq(r) = 1; end

    % ---------- inequalities ----------
    Ai = spalloc(0,nvar,0); bi = zeros(0,1);

    % z - w2 <= 0 ; w1 - z <= 0 ; sum_k w1 - t <= 0 ; w1 - w2 <= 0
    for m=1:M
        A3 = spalloc(nm(m)*K, nvar, 2*nm(m)*K);
        A4 = spalloc(nm(m)*K, nvar, 2*nm(m)*K);
        rr = 0;
        for ii=1:nm(m)
            zrow = idx.z{m}((ii-1)*K + (1:K));
            for k=1:K
                rr = rr+1;
                % z <= w2
                A3(rr, zrow(k))      =  1; A3(rr, idx.w2{m}(k)) = -1;
                % w1 <= z
                A4(rr, idx.w1{m}(k)) =  1; A4(rr, zrow(k))       = -1;
            end
        end
        Ai = [Ai; A3]; bi = [bi; zeros(nm(m)*K,1)];
        Ai = [Ai; A4]; bi = [bi; zeros(nm(m)*K,1)];

        % sum_k w1 - t <= 0   (keeps w1 not exceeding t)
        A5 = sparse(1,nvar); A5(1, idx.w1{m}) = 1; A5(1, idx.t) = -1;
        Ai = [Ai; A5]; bi = [bi; 0];

        % w1 - w2 <= 0  (ω' ≤ ω'')
        A6 = spalloc(K,nvar,2*K);
        for k=1:K, A6(k, idx.w1{m}(k)) = 1; A6(k, idx.w2{m}(k)) = -1; end
        Ai = [Ai; A6]; bi = [bi; zeros(K,1)];
    end

    % ---------- directional mDP constraints (within each cell) ----------
    [ii_all, jj_all, dirs] = find(neighborMatrix);
    localMap = cell(1,M);
    for m=1:M, localMap{m} = zeros(J,1); localMap{m}(hatX_groups{m}) = 1:nm(m); end
    for m=1:M
        rowsHere = 0;
        for rp=1:numel(ii_all)
            i=ii_all(rp); j=jj_all(rp);
            if localMap{m}(i)>0 && localMap{m}(j)>0, rowsHere=rowsHere+K; end
        end
        if rowsHere==0, continue; end
        A7 = spalloc(rowsHere,nvar,2*rowsHere); rr = 0;
        for rp=1:numel(ii_all)
            i=ii_all(rp); j=jj_all(rp); dcode=dirs(rp);
            iiLoc=localMap{m}(i); jjLoc=localMap{m}(j);
            if iiLoc==0 || jjLoc==0, continue; end
            eps_dir = (dcode==1)*epsilon1 + (dcode==2)*epsilon2;
            fac = min(exp(eps_dir * distanceMatrix(i,j)), 1e8);
            zi = idx.z{m}((iiLoc-1)*K + (1:K));
            zj = idx.z{m}((jjLoc-1)*K + (1:K));
            for k=1:K, rr=rr+1; A7(rr, zj(k)) = 1; A7(rr, zi(k)) = -fac; end
        end
        Ai = [Ai; A7]; bi = [bi; zeros(rowsHere,1)];
    end

    % ---------- tie shared anchors across cells ----------
    [anc, cel] = find(membershipCorners);
    cellLists  = accumarray(anc, cel, [J,1], @(v){v});
    for a=1:J
        cells = cellLists{a}; if numel(cells)<=1, continue; end
        localIdx = zeros(numel(cells),1);
        for tCell=1:numel(cells)
            m=cells(tCell); gi=hatX_groups{m};
            localIdx(tCell)=find(gi==a,1);
        end
        m0=cells(1); i0=localIdx(1); z0=idx.z{m0}((i0-1)*K + (1:K));
        Aadd = spalloc((numel(cells)-1)*K, nvar, 2*(numel(cells)-1)*K); rr=0;
        for tCell=2:numel(cells)
            mt=cells(tCell); it=localIdx(tCell); zt=idx.z{mt}((it-1)*K + (1:K));
            for k=1:K, rr=rr+1; Aadd(rr, zt(k))=1; Aadd(rr, z0(k))=-1; end
        end
        Aeq = [Aeq; Aadd]; beq = [beq; zeros(size(Aadd,1),1)];
    end

    % ---------- ε-tight min/max ----------
    for m=1:M
        nm_m = nm(m); if nm_m==0, continue; end
        S = spalloc(K, nvar, nm_m*K);
        for k=1:K, for ii=1:nm_m, S(k, idx.z{m}((ii-1)*K + k)) = 1; end, end
        % MIN tightening: sum_i z - nm*w1 <= (nm-1)*δ_min
        A_minT = S;
        for k=1:K, A_minT(k, idx.w1{m}(k)) = A_minT(k, idx.w1{m}(k)) - nm_m; end
        Ai = [Ai; A_minT]; bi = [bi; ones(K,1)*((nm_m-1)*deltaMin)];
        % MAX tightening: nm*w2 - sum_i z <= (nm-1)*δ_max
        A_maxT = -S;
        for k=1:K, A_maxT(k, idx.w2{m}(k)) = A_maxT(k, idx.w2{m}(k)) + nm_m; end
        Ai = [Ai; A_maxT]; bi = [bi; ones(K,1)*((nm_m-1)*deltaMax)];
    end

    % ---------- solve ----------
    opts = optimoptions('linprog','Display','none','Algorithm','interior-point');
    [x_opt, fval, exitflag, output] = linprog(f, Ai, bi, Aeq, beq, lb, ub, opts);

    % ---------- unpack ----------
    sol.exitflag = exitflag; sol.output = output; sol.fval = fval;
    sol.hatX_groups = hatX_groups;
    sol.Z_tilde = cell(1,M); sol.Z = cell(1,M);
    sol.omega_p_tilde = zeros(M,K); sol.omega_pp_tilde = zeros(M,K);
    sol.omega_p = zeros(M,K);       sol.omega_pp       = zeros(M,K);
    sol.t = NaN;

    if exitflag ~= 1, sol.status = sprintf('linprog_exitflag_%d', exitflag); return; end
    sol.status = 'ok';

    t = x_opt(idx.t); sol.t = t;
    for m=1:M
        zblock = x_opt(idx.z{m}); Zt = reshape(zblock, [K, nm(m)]).';
        sol.Z_tilde{m} = Zt;
        sol.omega_p_tilde(m,:)  = x_opt(idx.w1{m}).';
        sol.omega_pp_tilde(m,:) = x_opt(idx.w2{m}).';
        if t > 0
            sol.Z{m}         = Zt ./ t;
            sol.omega_p(m,:)  = sol.omega_p_tilde(m,:)  ./ t;  % lower bound (normalized)
            sol.omega_pp(m,:) = sol.omega_pp_tilde(m,:) ./ t;  % upper bound (normalized)
        else
            sol.Z{m} = NaN(nm(m),K);
            sol.omega_p(m,:)  = NaN(1,K);
            sol.omega_pp(m,:) = NaN(1,K);
        end
    end
end
