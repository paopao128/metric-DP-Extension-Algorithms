function print_loss_table(city_name, loss_em, loss_laplace, loss_tem, ...
                          loss_rmp, loss_copt, loss_lp, ...
                          loss_aipor, loss_bound, loss_aipo)

    % EPSILON values (assuming you used 0.2, 0.4, ..., 1.6)
    EPSILON_MAX = size(loss_em, 1);
    eps_vals = 0.2 * (1:EPSILON_MAX);

    % Pack methods in order
    method_names = { ...
        'EM', ...          % Pre-defined Noise
        'Laplace', ...
        'TEM', ...
        'RMP', ...         % Hybrid
        'COPT', ...
        'LP', ...          % Others
        'AIPO-R', ...
        'LB', ...
        'AIPO*'};

    loss_mats = { ...
        loss_em, ...
        loss_laplace, ...
        loss_tem, ...
        loss_rmp, ...
        loss_copt, ...
        loss_lp, ...
        loss_aipor, ...
        loss_bound, ...
        loss_aipo};

    pm = char(177);   % ± symbol

    %% Header
    fprintf('\n=============================================\n');
    fprintf('%s road map\n', city_name);
    fprintf('=============================================\n');

    % Column header
    fprintf('%-10s', 'Method');
    for i = 1:EPSILON_MAX
        fprintf('   e=%.1f', eps_vals(i));
    end
    fprintf('\n');
    fprintf(repmat('-', 1, 10 + 9 * EPSILON_MAX)); fprintf('\n');

    %% Pre-defined Noise Distribution
    fprintf('Pre-defined Noise Distribution:\n');
    for m = 1:3
        print_row(method_names{m}, loss_mats{m}, pm);
    end
    fprintf('\n');

    %% Hybrid Method
    fprintf('Hybrid Method:\n');
    for m = 4:5
        print_row(method_names{m}, loss_mats{m}, pm);
    end
    fprintf('\n');

    %% Other methods: LP, AIPO-R, LB, AIPO*
    for m = 6:9
        print_row(method_names{m}, loss_mats{m}, pm);
    end
    fprintf('\n');
end

%% Helper to print one method row
function print_row(name, loss_mat, pm)
    [EPSILON_MAX, ~] = size(loss_mat);
    mu = mean(loss_mat, 2);
    sd = std(loss_mat, 0, 2);

    fprintf('%-10s', name);
    for i = 1:EPSILON_MAX
        % Example: "  8.71±0.78"
        fprintf('  %5.2f%c%4.2f', mu(i), pm, sd(i));
    end
    fprintf('\n');
end