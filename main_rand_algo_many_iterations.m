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
%% Constructing the randomized reduced model: Uniform Sampling
profile on
n_rand = 10;
kt = 13;
time_steps_rand = 15;
rand_tol = 10^(-8);
prob = 0;

iterations = 10^5;
L2_errors_over_time_relative = zeros(iterations,N_time_steps+1);
L2_errors_max_relative = zeros(iterations,1);
rand_basis_dim = zeros(iterations,1);

for j = 1:iterations
    [rand_red_basis,rand_eigenvalues] = rand_red_basis_generation(n_rand,time_steps_rand,kt,prob,rand_tol,time_grid,u_0,M_bc,A_bc,rhs_matrix);
    reduced_mass = rand_red_basis'*M_bc*rand_red_basis;
    reduced_stiffness = rand_red_basis'*A_bc*rand_red_basis;
    reduced_rhs_matrix = rand_red_basis'*rhs_matrix;
    reduced_U_0 = ((reduced_mass)^(-1))*rand_red_basis'*M_bc*u_0;

    rand_basis_dim(j) = size(reduced_mass,2);
    % perform implicit euler on reduced model
    reduced_sol = euler_method(reduced_U_0,T_start,T_end,N_time_steps,reduced_mass,reduced_stiffness,reduced_rhs_matrix);

    [sz_red_x,sz_red_y] = size(reduced_sol);

    % generates reduced solution in FE space
    reduced_sol_for_error = rand_red_basis*reduced_sol;

    % Comparing the reduced model with the hi-fi model

    L2_errors_over_time = zeros(1,sz_red_y); % or eqiuvalently size of time grid
    L2_norms_FE_space = zeros(1,sz_red_y);

    for i = 1:sz_red_y
        error_temp = hf_solution(:,i) - reduced_sol_for_error(:,i);
        L2_errors_over_time(i) = sqrt(error_temp'*M_bc*error_temp);
        L2_norms_FE_space(i) = sqrt(hf_solution(:,i)'*M_bc*hf_solution(:,i));
        L2_errors_over_time_relative(j,i) = L2_errors_over_time(i)/L2_norms_FE_space(i);
    end
L2_errors_max_relative(j) = max(L2_errors_over_time_relative(j,:),[],'all');
end

% For making a table of the quantiles!

uniform_quants = quantile(L2_errors_over_time_relative,[0.05 0.25 0.5 0.75 0.9 0.95 0.97 0.98],'all');
uniform_quants = [min(L2_errors_over_time_relative,[],'all');uniform_quants;max(L2_errors_over_time_relative,[],'all')];
%% Constructing the randomized reduced model: Squared norm sampling

prob = 1;
L2_errors_over_time_relative_sq_norm = zeros(iterations,N_time_steps+1);
L2_errors_max_relative_sq_norm = zeros(iterations,1);
rand_basis_dim_sq_norm = zeros(iterations,1);

for j = 1:iterations
    [red_bas_sq_norm,eigs_sq_norm] = rand_red_basis_generation(n_rand,time_steps_rand,kt,prob,rand_tol,time_grid,u_0,M_bc,A_bc,rhs_matrix);
    red_mass_sq_norm = red_bas_sq_norm'*M_bc*red_bas_sq_norm;
    red_stiffs_sq_norm = red_bas_sq_norm'*A_bc*red_bas_sq_norm;
    red_rhs_sq_norm = red_bas_sq_norm'*rhs_matrix;
    red_U_0_sq_norm = ((red_mass_sq_norm)^(-1))*red_bas_sq_norm'*M_bc*u_0;

    rand_basis_dim_sq_norm(j) = size(red_mass_sq_norm,2);
    % perform implicit euler on reduced model
    red_sol_sq_norm = euler_method(red_U_0_sq_norm,T_start,T_end,N_time_steps,red_mass_sq_norm,red_stiffs_sq_norm,red_rhs_sq_norm);

    [sz_red_sq_norm_x,sz_red_sq_norm_y] = size(red_sol_sq_norm);

    % generates reduced solution in FE space
    red_sol_sq_norm_for_error = red_bas_sq_norm*red_sol_sq_norm;

    % Comparing the reduced model with the hi-fi model

    L2_errors_over_time_sq_norm = zeros(1,sz_red_sq_norm_y); % or eqiuvalently size of time grid
    L2_norms_FE_space_sq_norm = zeros(1,sz_red_sq_norm_y);

    for i = 1:sz_red_sq_norm_y
        error_temp = hf_solution(:,i) - red_sol_sq_norm_for_error(:,i);
        L2_errors_over_time_sq_norm(i) = sqrt(error_temp'*M_bc*error_temp);
        L2_norms_FE_space_sq_norm(i) = sqrt(hf_solution(:,i)'*M_bc*hf_solution(:,i));
        L2_errors_over_time_relative_sq_norm(j,i) = L2_errors_over_time_sq_norm(i)/L2_norms_FE_space_sq_norm(i);
    end
L2_errors_max_relative_sq_norm(j) = max(L2_errors_over_time_relative_sq_norm(j,:),[],'all');
end

% For making a table of the quantiles!

sq_norm_quants = quantile(L2_errors_over_time_relative_sq_norm,[0.05 0.25 0.5 0.75 0.9 0.95 0.97 0.98],'all');
sq_norm_quants = [min(L2_errors_over_time_relative_sq_norm,[],'all');sq_norm_quants;max(L2_errors_over_time_relative_sq_norm,[],'all')];

%% Randomized RB generation: rank-3 leverage score sampling

prob = 2;
L2_errors_over_time_relative_lev_score = zeros(iterations,N_time_steps+1);
L2_errors_max_relative_lev_score = zeros(iterations,1);
rand_basis_dim_lev_score = zeros(iterations,1);

for j = 1:iterations
    [red_bas_lev_score,eigs_lev_score] = rand_red_basis_generation(n_rand,time_steps_rand,kt,prob,rand_tol,time_grid,u_0,M_bc,A_bc,rhs_matrix);
    red_mass_lev_score = red_bas_lev_score'*M_bc*red_bas_lev_score;
    red_stiffs_lev_score = red_bas_lev_score'*A_bc*red_bas_lev_score;
    red_rhs_lev_score = red_bas_lev_score'*rhs_matrix;
    red_U_0_lev_score = ((red_mass_lev_score)^(-1))*red_bas_lev_score'*M_bc*u_0;

    rand_basis_dim_lev_score(j) = size(red_mass_lev_score,2);
    % perform implicit euler on reduced model
    red_sol_lev_score = euler_method(red_U_0_lev_score,T_start,T_end,N_time_steps,red_mass_lev_score,red_stiffs_lev_score,red_rhs_lev_score);

    [sz_red_lev_score_x,sz_red_lev_score_y] = size(red_sol_lev_score);

    % generates reduced solution in FE space
    red_sol_lev_score_for_error = red_bas_lev_score*red_sol_lev_score;

    % Comparing the reduced model with the hi-fi model

    L2_errors_over_time_lev_score = zeros(1,sz_red_lev_score_y); % or eqiuvalently size of time grid
    L2_norms_FE_space_lev_score = zeros(1,sz_red_lev_score_y);

    for i = 1:sz_red_lev_score_y
        error_temp = hf_solution(:,i) - red_sol_lev_score_for_error(:,i);
        L2_errors_over_time_lev_score(i) = sqrt(error_temp'*M_bc*error_temp);
        L2_norms_FE_space_lev_score(i) = sqrt(hf_solution(:,i)'*M_bc*hf_solution(:,i));
        L2_errors_over_time_relative_lev_score(j,i) = L2_errors_over_time_lev_score(i)/L2_norms_FE_space_lev_score(i);
    end
L2_errors_max_relative_lev_score(j) = max(L2_errors_over_time_relative_lev_score(j,:),[],'all');
end

% For making a table of the quantiles!

lev_score_quants = quantile(L2_errors_over_time_relative_lev_score,[0.05 0.25 0.5 0.75 0.9 0.95 0.97 0.98],'all');
lev_score_quants = [min(L2_errors_over_time_relative_lev_score,[],'all');lev_score_quants;max(L2_errors_over_time_relative_lev_score,[],'all')];
profile viewer
%% Making the table of quantiles!

table_sz = [size(uniform_quants,1) 4];
varTypes = {'double','double','double','double'};
varNames = {'Quantiles', 'Uniform Sampling','Squared Norm Sampling','Rank-3 Leverage Score Sampling'};
T = table('Size',table_sz,'VariableTypes',varTypes,'VariableNames',varNames);
quantiles = [0 0.05 0.25 0.5 0.75 0.9 0.95 0.97 0.98 1];
format long
for i = 1:table_sz(1)
    T(i,:) = {quantiles(i),uniform_quants(i),sq_norm_quants(i),lev_score_quants(i)};
end
disp(T);