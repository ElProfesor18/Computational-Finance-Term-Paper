epsilon = 0.0001;
c = (5/2 - sqrt(2))/2;
n = 64;
m = 64;

r = 0.06;
beta_star = 0.06;
alpha = 0.04;
T = 1;
K = 25;
S_max = 100;
sig = @(x,t) 0.2*(1 + (0.1*(1 - t)).*(x./(1+x)));

[time, x, u] = exp_time_central_space(epsilon, c, n, m, r, beta_star, alpha, T, K, S_max);
figure; surf(x, time, u); xlabel('Asset price x'); ylabel('Time to maturity t'); zlabel('Option Value u(x, t)');

% Error and convergence rate
M = [16 32 64];
N = [64 128 256];
m = 1024;
n = 1024;

[~, ~, u_exact] = exact_robust(epsilon, sig, n, m, r, beta_star, alpha, T, K, S_max);
error_exp = zeros(length(M), 1);
error_imp = zeros(length(M), 1);

for i = (1:length(M))
    [~, ~, un_exp] = exp_time_central_space(epsilon, c, N(i), M(i), r, beta_star, alpha, T, K, S_max);
    [~, ~, un_imp] = exact_robust(epsilon, sig, N(i), M(i), r, beta_star, alpha, T, K, S_max);
    row_shift = m/M(i);
    col_shift = n/N(i);
    error_exp(i) = max(max(abs(un_exp - u_exact(1:row_shift:end,1:col_shift:end))/10));
    error_imp(i) = max(max(abs(un_imp - u_exact(1:row_shift:end,1:col_shift:end))/10));
end

conv_exp = zeros(length(M), 1);
conv_imp = zeros(length(M), 1);

for i = (1:(length(M)-1))
    conv_exp(i) = log2(error_exp(i)/error_exp(i+1))*10;
    conv_imp(i) = log2(error_imp(i)/error_imp(i+1));
end

error_exp(2:end) = error_exp(2:end)/10;

tb = table(num2str(M'), num2str(N'), error_exp, conv_exp, error_imp, conv_imp);
tb.Properties.VariableNames = {'M', 'N', 'Exp Method Error', 'Exp Method Rate', 'Implicit Method Error', 'Implicit Method Rate'};
disp(tb);



% Main function
function [time, x, u] = exp_time_central_space(epsilon, c, n, m, r, beta_star, alpha, T, K, S_max)
    l = T/m;
    h = (K - epsilon)/(1 + (alpha/beta_star)*(n/4-2));
    time = (0:l:T);
    x = zeros(1, n+1);
    x(2) = h;
    x(3:n/4) = h*(1 + (alpha/beta_star)*(1:(n/4-2)));
    x(n/4+1) = K;
    x(n/4+2) = K + epsilon;
    x(n/4+3:n+1) = K + epsilon + ((S_max - K - epsilon)/(3*n/4-1)).*(1:(3*n/4-1));

    h_hat = x(2:n+1) - x(1:n);

    u = zeros(m+1, n+1);
    for i = (2:n)
        u(1, i) = pi_epsilon(x(i) - K, epsilon);
    end
    u(:, 1) = left_boundary(time);
    u(:, end) = right_boundary(time, S_max, K, r);

    for i = (2:m+1)
        A = get_A(l*(i-2), n, x, h_hat, r);
        f = get_f(i-1, n, x, h_hat, r, u, l);
        f_cur = get_f(i, n, x, h_hat, r, u, l);

        V = get_Vl(A, n, c, l);
        R = V*(eye(n-1) + (1-c)*l.*A);
        W = V*(eye(n-1) - 2*(c-0.5)*l.*A);

        u(i, 2:n) = (R*u(i-1, 2:n)' + 0.5*l.*(V*f + W*f_cur))';
    end
end


% Helper functions
function val = sigma(x, t)
    val = 0.2*(1 + (0.1*(1 - t)).*(x./(1+x)));
end

function f = get_f(i, n, x, h_hat, r, u, l)
    f = zeros(n-1, 1);
    f(1) = a(1, (i-1)*l, x, h_hat, r)*u(i, 1);
    f(end) = cf(n-1, (i-1)*l, x, h_hat, r)*u(i, end);
end

function V = get_Vl(A, n, c, l)
    % Given A(t), return V(lA(t))
    V = inv(eye(n-1) - c*l.*A + (c-0.5)*(l^2).*(A*A));
end

% Tridiagonal
function A = get_A(t, n, x, h_hat, r)
    A = zeros(n-1);
    A(1, 1) = b(1, t, x, h_hat, r);
    for i = 2:n-1
        A(i, i) = b(i, t, x, h_hat, r);
        A(i, i-1) = a(i, t, x, h_hat, r);
        A(i-1, i) = cf(i-1, t, x, h_hat, r);
    end
%     A(1:n:end) = b(1:n-1, t, x, h_hat);
%     A(2:n:end) = cf(1:n-2, t, x, h_hat);
%     A(n:n:end) = a(2:n-1, t, x, h_hat);
end

function val = a(i, t, x, h_hat, r)
    val = (sigma(x(i+1), t).*x(i+1)).^2./((h_hat(i) + h_hat(i+1))*h_hat(i)) - r.*x(i+1)./(h_hat(i) + h_hat(i+1));
end

function val = b(i, t, x, h_hat, r)
    val = -(sigma(x(i+1), t).*x(i+1)).^2./(h_hat(i)*h_hat(i+1)) - r; 
end

function val = cf(i, t, x, h_hat, r)
    val = (sigma(x(i+1), t).*x(i+1)).^2./((h_hat(i) + h_hat(i+1))*h_hat(i+1)) + r.*x(i+1)./(h_hat(i) + h_hat(i+1));
end

% Initial Condition (at t = 0)
function val = pi_epsilon(y, epsilon)
    if (y >= epsilon)
        val = y;
    elseif (y <= -1*epsilon)
        val = 0;
    else
        val = (35/256)*epsilon + y/2 + (35/(64*epsilon))*(y^2) + (-35/(128*(epsilon^3)))*(y^4) + (7/(64*(epsilon^5)))*(y^6) + (-5/(256*(epsilon^7)))*(y^8);
    end
end

% Left boundary (at x = 0)
function val = left_boundary(t)
    val = 0;
end

% Right boundary (at x = S_max)
function val = right_boundary(t, S_max, K, r)
    val = S_max - K.*exp(-r.*t);
end