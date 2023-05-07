function [A_bc,M_bc,F,x_grid,y_grid] = FE_adv_diff_assemble(a,b,f_x,f_y,n_x,n_y,x0,x1,y0,y1)
    
% this will assemble a FE space in two spatial dimensions for the
% advection-diffusion problem, which has the diff eq
    % (- a Laplacian u) + (b dot grad(u)) = f

% our domain is D = [x0,y0]*[y0,y1], with homogeneous Dirichlet BCs

% b should be passed as a two-entry array, i.e. b = [b_x b_y];

% discretizing the spatial domain D

h_x = 1/n_x;
h_y = 1/n_y;

x_grid = [x0:h_x:x1]';
y_grid = [y0:h_y:y1]';

tolquad = 1e-12;

% assembling in x- and y-directions:

[A_bc_x,M_bc_x,F_x,~] = FE_adv_diff_1D(n_x,a,b(1),f_x,x0,x1);
[A_bc_y,M_bc_y,F_y,~] = FE_adv_diff_1D(n_y,a,b(2),f_y,y0,y1);

% assembling the matrices for the 2D space:

A_bc = kron(A_bc_x,M_bc_y) + kron(M_bc_x,A_bc_y);
M_bc = kron(M_bc_x,M_bc_y);

% assembling the RHS vector:

f_num = size(f_x,1);
F = zeros(size(F_x,1)*size(F_y,1),f_num);

for j = 1:f_num
    F(:,j) = kron(F_x(:,j),F_y(:,j));
end

% F = sum(F,2) - DON'T USE YET
% with the outputs, we will be able to solve the system A_bc*U = F
% the mass matrix is needed for the Euler method

end