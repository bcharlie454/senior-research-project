%% This will test the deterministic basis generation algorithm involving the transfer operator, to make sure it works!

clear all
figure(1)
clf
figure(2)
clf
figure(3)
clf

%% Defining the stoves, initial conditions, and other initial data

% time functions for the stoves
f_t = cell(3,1);
f_t{1,1} = @(t) 10.*(-t.^2+4.5.*t-2).*(t>=0.5).*(t<=4);
f_t{2,1} = @(t) 5.*(-t.^2+10.*t-21).*(t>=3).*(t<=7);
f_t{3,1} = @(t) 10.*(-t.^2+15.*t-54).*(t>=6).*(t<=9);

% where the stoves are in the x-domain
f_x = cell(3,1);
f_x{1,1} = @(x) (x>=0.2).*(x<=0.3);
f_x{2,1} = @(x) (x>=0.45).*(x<=0.55);
f_x{3,1} = @(x) (x>=0.65).*(x<=0.8);

% where the stoves are in the y-domain
f_y = cell(3,1);
f_y{1,1} = @(y) (y>=0.2).*(y<=0.3);
f_y{2,1} = @(y) (y>=0.45).*(y<=0.55);
f_y{3,1} = @(y) (y>=0.65).*(y<=0.8);

% setting up the grid, and time points
n_x = 100;
n_y = 100;
h_x = 1/n_x;
h_y = 1/n_y;
grid_x = [0:h_x:1]';
grid_y = [0:h_y:1]';
sz_x = size(grid_x,1);
sz_y = size(grid_y,1);
[X,Y] = meshgrid(grid_x,grid_y);

T_start = 0;
T_end = 10;
step_size = 1/30;
N_time_steps = (T_end-T_start)/step_size;
time_grid = [T_start:step_size:T_end]';

% initial conditions
u_0 = @(x,y) sin(pi*x)*sin(pi*y) + sin(2*pi*x)*sin(2*pi*y) + sin(3*pi*x)*sin(3*pi*y);
u_0 = u_0(X,Y);
u_0 = u_0(:,2:end-1);
u_0 = u_0(2:end-1,:);
u_0 = reshape(u_0,[],1);

%% Computing the high-fidelity solution

% creating the FE space and the righthand-side matrix
[A_bc,M_bc,F,grid_x,grid_y]=FE_assembly(n_x,n_y,f_x,f_y);

first_time = f_t{1,1}(time_grid);
second_time = f_t{2,1}(time_grid);
third_time = f_t{3,1}(time_grid);

% this code makes it so that the right-hand side of the matrix is related
% to the source terms \sum_{i=1}^3 f_i(t)f_i(x,y)
rhs_matrix = zeros(size(F,1),size(F,2),size(time_grid,1));
for i=1:size(time_grid,1)
    rhs_matrix(:,1,i) = F(:,1).*first_time(i);
    rhs_matrix(:,2,i) = F(:,2).*second_time(i);
    rhs_matrix(:,3,i) = F(:,3).*third_time(i);
end

rhs_matrix = sum(rhs_matrix,2);
rhs_matrix = reshape(rhs_matrix,size(F,1),size(time_grid,1));
 
hf_solution = euler_method(u_0,T_start,T_end,N_time_steps,M_bc,A_bc,rhs_matrix);

%% Reduced basis generation which incorporates essentially all high-fidelity data

nt_loc = 30;
kt_loc = 1;
tol = 10^(-8);

[eigenvectors,eigenvalues,singular_values,reduced_basis,reduced_dimension,tol_val,energy] = transfer_operator_deterministic(M_bc,A_bc,rhs_matrix,hf_solution,time_grid,nt_loc,kt_loc,tol);

reduced_mass = reduced_basis'*M_bc*reduced_basis;
reduced_stiffness = reduced_basis'*A_bc*reduced_basis;
reduced_rhs_matrix = reduced_basis'*rhs_matrix;
reduced_U_0 = ((reduced_mass)^(-1))*reduced_basis'*M_bc*u_0;

% perform implicit euler on reduced model
reduced_sol = euler_method(reduced_U_0,T_start,T_end,N_time_steps,reduced_mass,reduced_stiffness,reduced_rhs_matrix);

[sz_red_x,sz_red_y] = size(reduced_sol);

% generates reduced solution in FE space
reduced_sol_for_error = reduced_basis*reduced_sol;

%% Comparing the reduced model with the hi-fi model

L2_errors_over_time = zeros(1,sz_red_y); % or eqiuvalently size of time grid
L2_norms_FE_space = zeros(1,sz_red_y);
L2_errors_over_time_relative = zeros(1,sz_red_y);

for i = 1:sz_red_y
    error_temp = hf_solution(:,i) - reduced_sol_for_error(:,i);
    L2_errors_over_time(i) = sqrt(error_temp'*M_bc*error_temp);
    L2_norms_FE_space(i) = sqrt(hf_solution(:,i)'*M_bc*hf_solution(:,i));
    L2_errors_over_time_relative(i) = L2_errors_over_time(i)/L2_norms_FE_space(i);
end

%% Constructing Reduced Model with Standard POD, where nt_POD = 301

nt_POD = 301;
[sing_val_POD, red_basis_POD, eigenvectors_POD, eigenvals_POD, tol_val_POD, red_dim_POD, energy_POD] = standard_POD(M_bc, hf_solution, nt_POD, tol);
red_mass_POD = red_basis_POD'*M_bc*red_basis_POD;
red_stiffs_POD = red_basis_POD'*A_bc*red_basis_POD;
red_rhs_POD = red_basis_POD'*rhs_matrix;
red_U_0_POD = ((red_mass_POD)^(-1))*red_basis_POD'*M_bc*u_0;

% perform implicit euler on reduced model
red_sol_POD = euler_method(red_U_0_POD,T_start,T_end,N_time_steps,red_mass_POD,red_stiffs_POD,red_rhs_POD);

[sz_red_x_POD,sz_red_y_POD] = size(red_sol_POD);

% generates reduced solution in FE space
red_sol_POD_for_error = red_basis_POD*red_sol_POD;

%% Comparing the reduced model with the hi-fi model

L2_errors_over_time_POD = zeros(1,sz_red_y_POD); % or eqiuvalently size of time grid
L2_norms_FE_space_POD = zeros(1,sz_red_y_POD);
L2_errors_over_time_relative_POD = zeros(1,sz_red_y_POD);

for i = 1:sz_red_y_POD
    error_temp = hf_solution(:,i) - red_sol_POD_for_error(:,i);
    L2_errors_over_time_POD(i) = sqrt(error_temp'*M_bc*error_temp);
    L2_norms_FE_space_POD(i) = sqrt(hf_solution(:,i)'*M_bc*hf_solution(:,i));
    L2_errors_over_time_relative_POD(i) = L2_errors_over_time_POD(i)/L2_norms_FE_space_POD(i);
end

%% Plotting

figure(1)
p = semilogy(time_grid,L2_errors_over_time_relative); 
grid on
p.LineWidth = 1;
hold on
p1 = semilogy(time_grid,L2_errors_over_time_relative_POD); 
grid on
p1.LineWidth = 1.5;
p1.Color = '#006600';
p1.LineStyle = "--";
xlabel('$t$','Interpreter','latex');
ylabel('Relative $L^2(t)$ errors over time','Interpreter','latex');
legend('Deterministic Local Solver','Standard POD');

figure(2)
q = semilogy(tol_val);
grid on
q.LineStyle = "--";
q.Marker = ".";
q.Color = "b";
q.LineWidth = 1;
q.MarkerSize = 12;
hold on
q1 = semilogy(tol_val_POD);
q1.LineStyle = "-.";
q1.Marker = "*";
q1.Color = '#006600';
q1.LineWidth = 1;
q1.MarkerSize = 12;
hold on 
yline(10^-8,'r-','LineWidth',1);
xlabel('Singular value number');
ylabel('Singular value error ratio');
legend('Deterministic Local Solver','Standard POD','Tolerance');

figure(3)
first_eigenvals = eigenvalues(1:reduced_dimension);
pl = semilogy(first_eigenvals);
grid on
pl.LineStyle = "--";
pl.Marker = ".";
pl.Color = "b";
pl.LineWidth = 1;
pl.MarkerSize = 12;
hold on
first_eigenvals = eigenvals_POD(1:red_dim_POD);
pl1 = semilogy(first_eigenvals);
pl1.LineStyle = "-.";
pl1.Marker = "*";
pl1.Color = '#006600';
pl1.LineWidth = 1;
pl1.MarkerSize = 12;
xlabel('Eigenvalue number');
ylabel('Eigenvalue');
legend('Deterministic Local Solver','Standard POD');