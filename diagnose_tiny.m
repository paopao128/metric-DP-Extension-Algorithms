%% Step-by-step: what happens when x(i,k) = 1×J vector?
J=3; nFine=4; K=2;
rng(1);
z = rand(J, K);   z = z ./ sum(z);
W = rand(nFine, J);  W = W ./ sum(W,2);

fprintf('z(:,1) = %s\n', mat2str(z(:,1)',4));
fprintf('W(1,:) = %s\n', mat2str(W(1,:),4));

a = z(:,1)';      % 1×J
b = W(1,:)';      % J×1
ab = a .^ b;
p = prod(ab);
fprintf('\na .^ b: size=%s\n', mat2str(size(ab)));
fprintf('prod(a .^ b): size=%s,  val=%s\n', mat2str(size(p)), mat2str(p,4));

%% Step through assignment manually
fprintf('\n--- First iteration (k=1, i=1): x(1,1) = prod(J×J) ---\n');
clear x_test;
k=1; i=1;
val = prod(z(:,k)' .^ W(i,:)');
fprintf('val size = %s,  val = %s\n', mat2str(size(val)), mat2str(val,4));
x_test(i,k) = val;
fprintf('After x_test(1,1) = val:\n');
fprintf('  size(x_test) = %s\n', mat2str(size(x_test)));
fprintf('  x_test = %s\n', mat2str(x_test,4));

fprintf('\n--- Second iteration (k=1, i=2) ---\n');
i=2;
val2 = prod(z(:,k)' .^ W(i,:)');
fprintf('val2 = %s\n', mat2str(val2,4));
x_test(i,k) = val2;
fprintf('After x_test(2,1) = val2:\n');
fprintf('  size(x_test) = %s\n', mat2str(size(x_test)));
fprintf('  x_test = \n'); disp(x_test);

%% Now check: what is the CORRECT log-convex value for i=1, k=1?
correct_val = prod(z(:,1) .^ W(1,:)');  % scalar
fprintf('Correct log-convex (scalar formula): %.6f\n', correct_val);
fprintf('prod(prod(a .^ b)):                  %.6f\n', prod(prod(a .^ b)));
fprintf('prod(a) .^ sum(b):                   %.6f\n', prod(a) .^ sum(b));

%% Full loop
clear x_A;
for k=1:K
    for i=1:nFine
        x_A(i,k) = prod(z(:,k)' .^ W(i,:)');
    end
end
fprintf('\nFull x_A (size %s):\n', mat2str(size(x_A)));
disp(x_A);

%% True log-convex reference
x_ref = zeros(nFine,K);
for k=1:K
    for i=1:nFine
        x_ref(i,k) = prod(z(:,k) .^ W(i,:)');
    end
end
fprintf('True log-convex x_ref (size %s):\n', mat2str(size(x_ref)));
disp(x_ref);
fprintf('max diff x_A vs x_ref = %.2e\n', max(abs(x_A(:)-x_ref(:))));
