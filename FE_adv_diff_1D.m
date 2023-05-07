function [A_bc,M_bc,F,x_grid] = FE_adv_diff_1D(n,a,b,f,x0,x1)

% this performs FE spatial discretization for the advection-diffusion
% equation in 1D

% the spatial part of the equation is 
    % -au''(x) + bu'(x) = f(x)

% where x is in (x0,x1)

% we prescribe homogeneous Dirichlet BCs, i.e. u(x0) = u(x1) = 0
    
% the negative sign is important!!! Otherwise we get funky results

% in this code, a and b must be constants
% this will be generalized for spatially-dependent a and b later

tolquad = 1e-12;
h = 1/n;
x_grid = [x0:h:x1]';
n_for_matrices = (x1-x0)*n; % this ensures that the matrices are scaled in the correct manner based on the non-unit interval

% assembling the stiffness matrix for homogeneous Dirichlet BCs

e = ones((n_for_matrices-1),1);
A_temp = spdiags([(-a - h*b/2).*e (2*a).*e (-a + h*b/2).*e],-1:1,n_for_matrices-1,n_for_matrices-1);
A_bc = (1/h).*A_temp;

% assembling the mass matrix for homogeneous BCs

M_temp = spdiags([e 4.*e e],-1:1,n_for_matrices-1,n_for_matrices-1);
M_bc = (h/6).*M_temp;

% assembling the RHS vector 

% first we need to figure out if f is a sum of functions (e.g. f could be
% of the form f_1 + f_2 + f_3 + ...)

f_num = size(f,1);
F = zeros((n_for_matrices-1),f_num);

bas_hix = cell(2,1);
f_bas_prod = cell(2,f_num);
int_hix = zeros(f_num,1);

for i = 1:(n_for_matrices-1)

    % these are the FE basis functions
    bas_hix{1,1} = @(x) (x - (x0 + (i-1)*h))/h;
    bas_hix{2,1} = @(x) ((x0 + (i+1)*h) - x)/h;

    for j = 1:f_num
        % creating integrand for assembling the RHS vector
        f_bas_prod{1,j} = @(x) f{j,1}(x) .* bas_hix{1}(x);
        f_bas_prod{2,j} = @(x) f{j,1}(x) .* bas_hix{2}(x);

        % computing the integral of these integrands
        int_hix(j) = integral(f_bas_prod{1,j},(x0 + (i-1)*h),(x0 + i*h),'RelTol',tolquad) + integral(f_bas_prod{2,j},(x0 + i*h),(x0 + (i+1)*h),'RelTol',tolquad);

        % adding the integral expression to the RHS vector
        F(i,j) = F(i,j) + int_hix(j);
    end
end
end