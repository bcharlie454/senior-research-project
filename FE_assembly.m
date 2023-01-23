function [K_bc,M_bc,F,grid_x,grid_y]=FE_assembly(n_x,n_y,f_x,f_y)

% Specify tolerance for integration

tolquad = 1e-12;

h_x = 1/n_x;

grid_x = [0:h_x:1]';

h_y = 1/n_y;

grid_y = [0:h_y:1]';
             
% Assemble in x-direction and y-direction

[K_bc_x,M_bc_x,F_x]=FE_assembly_1d(n_x,f_x);
[K_bc_y,M_bc_y,F_y]=FE_assembly_1d(n_y,f_y);

% Build 2D matrices

K_bc = kron(K_bc_x,M_bc_y) + kron(M_bc_x,K_bc_y);

M_bc = kron(M_bc_x,M_bc_y);

f_num = size(f_x,1);

F = zeros(size(F_x,1)*size(F_y,1),f_num);

for lk = 1:f_num

  F(:,lk) = kron(F_x(:,lk),F_y(:,lk));

end

end