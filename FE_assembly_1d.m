function [K_bc,M_bc,F]=FE_assembly_1d(n,f)

% Specify tolerance for integration

tolquad = 1e-12;

h = 1/n;

grid = [0:h:1]';
             
% Stiffness matrix with homogeneous Dirichlet boundary conditions

e = ones(n-1,1);
aux = spdiags([-e 2.*e -e],-1:1,n-1,n-1);
K_bc = (1/h).*aux;
        
% Mass matrix with homogeneous boundary conditions

aux = spdiags([e 4.*e e],-1:1,n-1,n-1);
M_bc = (h/6).*aux;

% Matrices with Neumann b.c. for norm calculations

e1 = [1;2*e;1];
aux = spdiags([-ones(n+1,1) e1 -ones(n+1,1)],-1:1,n+1,n+1);
K = (1/h).*aux;

e = ones(n+1,1);
e1 = [1;2.*e(2:n);1];
M = spdiags([0.5.*e e1 0.5.*e],-1:1,n+1,n+1);
M = M.*(h/3);


% Assemble f

% How many summands in affine decomposition?

f_num = size(f,1);
F = zeros((n-1),f_num);

bas_hix = cell(2,1);
prod_f = cell(2,f_num);
int_hix = zeros(f_num,1);


for i = 1:n -1 
              
  % 1D FEM basis functions 
        
  bas_hix{1} = @(x) (x - ((i-1)*h))/h;
  bas_hix{2} = @(x) (((i + 1)*h) - x)/h;

  for lk = 1:f_num
    prod_f{1,lk} = @(x) f{lk}(x) .* bas_hix{1}(x);
    prod_f{2,lk} = @(x) f{lk}(x) .* bas_hix{2}(x);
    int_hix(lk) = integral(prod_f{1,lk},((i-1)*h),i*h,'RelTol',tolquad) + integral(prod_f{2,lk},(i*h),((i+1)*h),'RelTol',tolquad);
            
    % Add the integrals to the vector of the right-hand side.                 
                
    F(i,lk) = F(i,lk) + int_hix(lk);
    
  end
  
end


end