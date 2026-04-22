function print_violation_table(city_name, viol_em, viol_laplace, viol_tem, ...
                               viol_rmp, viol_copt, viol_lp, ...
                               viol_aipor, viol_aipo)
% Print violation ratios as mean±std for each epsilon

    EPSILON_MAX = size(viol_em, 1);
    eps_vals    = 0.2 * (1:EPSILON_MAX);   % adjust if needed

    method_names = { ...
        'EM', ...          % Pre-defined Noise
        'Laplace', ...
        'TEM', ...
        'RMP', ...         % Hybrid
        'COPT', ...
        'LP', ...          % Others
        'AIPO-R', ...
        'AIPO*'};

    viol_mats = { ...
        viol_em, ...
        viol_laplace, ...
        viol_tem, ...
        viol_rmp, ...
        viol_copt, ...
        viol_lp, ...
        viol_aipor, ...
        viol_aipo};

    pm = char(177);   % ± symbol

    fprintf('\n=============================================\n');
    fprintf('%s road map – Violation ratio\n', city_name);
    fprintf('=============================================\n');

    fprintf('%-10s', 'Method');
    for i = 1:EPSILON_MAX
        fprintf('   e=%.1f', eps_vals(i));
    end
    fprintf('\n');
    fprintf(repmat('-', 1, 10 + 9 * EPSILON_MAX)); fprintf('\n');

    % Pre-defined Noise Distribution
    fprintf('Pre-defined Noise Distribution:\n');
    for m = 1:3
        print_row_violation(method_names{m}, viol_mats{m}, pm);
    end
    fprintf('\n');

    % Hybrid
    fprintf('Hybrid Method:\n');
    for m = 4:5
        print_row_violation(method_names{m}, viol_mats{m}, pm);
    end
    fprintf('\n');

    % Others: LP, AIPO-R, AIPO*
    for m = 6:8
        print_row_violation(method_names{m}, viol_mats{m}, pm);
    end
    fprintf('\n');
end

function print_row_violation(name, viol_mat, pm)
    [EPSILON_MAX, ~] = size(viol_mat);
    mu = mean(viol_mat, 2);
    sd = std(viol_mat, 0, 2);

    fprintf('%-10s', name);
    for i = 1:EPSILON_MAX
        % e.g., "  0.12±0.03"
        fprintf('  %5.3f%c%4.3f', mu(i), pm, sd(i));
    end
    fprintf('\n');
end
