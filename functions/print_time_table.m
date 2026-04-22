function print_time_table(city_name, time_copt, time_lp, time_aipo)
% Print computation time (seconds) as mean±std for each epsilon

    EPSILON_MAX = size(time_copt, 1);
    eps_vals    = 0.2 * (1:EPSILON_MAX);   % adjust if needed

    method_names = { ...
        'COPT', ...
        'LP', ...          % Others
        'AIPO*'};

    time_mats = { ...
        time_copt, ...
        time_lp, ...
        time_aipo};

    pm = char(177);   % ± symbol

    fprintf('\n=============================================\n');
    fprintf('%s road map – Computation time (s)\n', city_name);
    fprintf('=============================================\n');

    fprintf('%-10s', 'Method');
    for i = 1:EPSILON_MAX
        fprintf('   e=%.1f', eps_vals(i));
    end
    fprintf('\n');
    fprintf(repmat('-', 1, 10 + 9 * EPSILON_MAX)); fprintf('\n');


    for m = 1:3
        print_row_time(method_names{m}, time_mats{m}, pm);
    end
    fprintf('\n');
end

function print_row_time(name, time_mat, pm)
    [EPSILON_MAX, ~] = size(time_mat);
    mu = mean(time_mat, 2);
    sd = std(time_mat, 0, 2);

    fprintf('%-10s', name);
    for i = 1:EPSILON_MAX
        % e.g., "  0.123±0.045"
        fprintf('  %7.3f%c%5.3f', mu(i), pm, sd(i));
    end
    fprintf('\n');
end
