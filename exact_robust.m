function [t, x, u] = exact_robust(epsilon, sigma, N, M, r, beta_star, alpha, T, K, S_max)
    bc1 = @(t) 0;
    bc2 = @(t) S_max - K * exp(-r*(1-t));
    
    x = zeros(1,N+1);
    h = zeros(1,N+1);
    for i = 2:N+1
        x(i) = get_x(i-1, N, K, alpha, beta_star, S_max, epsilon);
        h(i) = get_h(i-1, N, K, alpha, beta_star, S_max, epsilon);
    end
    t = (0:T/M:T);
    tau = T/M;
    u = zeros(N+1,M+1);
    u(1,:) = bc1(t);
    u(N+1,:) = bc2(t);
    for i = 2:N
        u(i,M+1) = pi_epsilon(x(i)-K,epsilon);
    end
    A = zeros(N+1,N+1);
    A(1,1) = 1;
    A(1,2) = 0;
    A(N+1,N) = 0;
    A(N+1,N+1) = 1;
    C = zeros(N+1,1);
    for j = M:-1:1
        for i = 2:N
            x_i = x(i);
            h_i = h(i);
            h_i_plus_1 = h(i+1);
            sig_ij = sigma(x_i,t(j))^2;
            r_j = r;
            A(i,i-1) = (tau * x_i) * (r_j - (sig_ij * x_i / (h_i))) / (h_i + h_i_plus_1);
            A(i,i) = 1 + (sig_ij  * tau * (x_i^2))/(h_i * h_i_plus_1) + r_j * tau;
            A(i,i+1) = -((tau * x_i) * (r_j + (sig_ij * x_i / (h_i_plus_1))) / (h_i + h_i_plus_1));
        end
        C(1) = u(1,j);
        C(N+1) = u(N+1,j);
        C(2:N) = u(2:N,j+1);
        u(:,j) = A\C;
    end
    u = u';
end


function val = pi_epsilon(y,epsilon)
    c_0 = (35/256)*epsilon; c_1 = 0.5 ; c_2 = 35/(64*epsilon); c_4 = -35/(128*epsilon^3);
    c_6 = 7/(64*epsilon^5); c_8 = -5/(256*epsilon^7);
    c_3 = 0;c_5=0;c_7=0;c_9=0;
    p = [c_9 c_8 c_7 c_6 c_5 c_4 c_3 c_2 c_1 c_0];
    if (y >= epsilon)
        val = y;
    elseif (y <= -epsilon) 
        val=0;
    else
        val=polyval(p, y);
    end
end

function x_i = get_x(i, N, K, alpha, beta_star, S_max, epsilon)
    h = (K - epsilon)/(1 + (alpha/beta_star)*(N/4 -2));
    if i == 1
        x_i = h;
    elseif i < N/4 && i>1
        x_i = h*(1 + (i-1)*alpha/beta_star);
    elseif i==(N/4)
        x_i = K;
    elseif i == (N/4 + 1)
        x_i = K + epsilon;
    else
        x_i = K + epsilon + (S_max - K - epsilon)*(i - N/4 - 1)/(3*N/4 - 1);
    end   
end

function h_i = get_h(i, N, K, alpha, beta_star, S_max, epsilon)
    h = (K-epsilon)/(1 + alpha*(N/4 - 2)/beta_star);
    if i==1
        h_i = h;
    elseif i < N/4 && i>1
        h_i = h*alpha/beta_star;
    elseif i < (N/4 + 2) && i>=N/4
        h_i = epsilon;
    else
        h_i = (S_max - K - epsilon)/(3*N/4 - 1);
    end   
end