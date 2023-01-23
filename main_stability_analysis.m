clear all

%% Setting the scene

% defining functions for source terms 

f_x = cell(1,1);
f_x{1,1} = @(x) 4*pi*sin(2*pi*x);
f_y = cell(1,1);
f_y{1,1} = @(y) 2*pi*sin(2*pi*y);

% Choose number of elements in x and y direction 

n_x = 500;
n_y = 500;

% Assemble matrices (with _bc means boundary conditions):

[A_bc,M_bc,F,grid_x,grid_y]=FE_assembly(n_x,n_y,f_x,f_y);


% Exact solution

u = @(x,y) sin(2*pi*x).*sin(2*pi*y); 

[X,Y] = meshgrid(grid_x,grid_y);
U_exact_in_space = u(X,Y);

% t = 1;
% U_exact_at_time_t = exp(-t).*U_exact_in_space; % NOTE: t needs to be specified
% 
% figure(1) 
% surf(X,Y,U_exact_at_time_t); shading interp

% Numerical solution of \partial_t u -\Delta u = f

% inputs for the Euler method and exact solution
T_start = 0;
T_end = 1;
time_steps = 10;
step_size = (T_end - T_start)/time_steps;
time_grid = [T_start:step_size:T_end]';
f_t = cell(4,1);
a = [-10,0.5,1.5,2.5]';
f_t{1,1} = @(t) exp(-a(1).*t);
f_t{2,1} = @(t) exp(-a(2).*t);
f_t{3,1} = @(t) exp(-a(3).*t);
f_t{4,1} = @(t) exp(-a(4).*t);
time_values = zeros(size(a,1),time_steps+1);

U_exact = zeros(size(U_exact_in_space,1),size(U_exact_in_space,2),size(a,1),time_steps+1);
% calculating the exact solution
for i=1:size(a,1)
    time_values(i,:) = f_t{i,1}(time_grid);
    for j = 1:time_steps+1
        U_exact(:,:,i,j) = time_values(i,j).*U_exact_in_space;
    end
end

% First assemble u_0:
u_0 = u(X,Y);
u_0 = u_0(:,2:end-1);
u_0 = u_0(2:end-1,:);
u_0 = reshape(u_0,[],1);

% evaluating the euler method for different time steps
rhs_matrix = zeros(size(F,1),time_steps+1,size(a,1));

for i = 1:time_steps+1
    for j = 1:size(a,1)
        rhs_matrix(:,i,j) = (1 - 1/(8*pi*pi)).*exp(-a(j).*(step_size)*(i-1)).*F;
    end
end

%% Numerical solution and error at 1st time step, 5th time step, and final time step (2, 6, 11)

numerical_solution = zeros(size(F,1),time_steps+1,size(a,1));

% computes the numerical solution for all values of a, then picks out
% specific time points
for i=1:size(a,1)
    numerical_solution(:,:,i) = euler_method(u_0,T_start,T_end,time_steps,M_bc,A_bc,rhs_matrix(:,:,i));
end

%% This is the unstable value of a
numer_sol_bad = numerical_solution(:,:,1);
% for time t = 0.1
U_vis_2_bad = reshape(numer_sol_bad(:,2),n_x-1,[])';
U_vis_2_bad = [zeros(n_x-1,1),U_vis_2_bad,zeros(n_x-1,1)];
U_vis_2_bad = [zeros(1,n_y+1);U_vis_2_bad;zeros(1,n_y+1)];
error_matrix_2_bad = abs(U_exact(:,:,1,2) - U_vis_2_bad);
max_error_2_bad = max(error_matrix_2_bad,[],'all');

% for time t = 0.5
U_vis_6_bad = reshape(numer_sol_bad(:,6),n_x-1,[])';
U_vis_6_bad = [zeros(n_x-1,1),U_vis_6_bad,zeros(n_x-1,1)];
U_vis_6_bad = [zeros(1,n_y+1);U_vis_6_bad;zeros(1,n_y+1)];
error_matrix_6_bad = abs(U_exact(:,:,1,6) - U_vis_6_bad);
max_error_6_bad = max(error_matrix_6_bad,[],'all');

% for time t = 1
U_vis_11_bad = reshape(numer_sol_bad(:,11),n_x-1,[])';
U_vis_11_bad = [zeros(n_x-1,1),U_vis_11_bad,zeros(n_x-1,1)];
U_vis_11_bad = [zeros(1,n_y+1);U_vis_11_bad;zeros(1,n_y+1)];
error_matrix_11_bad = abs(U_exact(:,:,1,11) - U_vis_11_bad);
max_error_11_bad = max(error_matrix_11_bad,[],'all');

% Next we move to the stable values of a
%% for a = 0.2
numer_sol_first_good = numerical_solution(:,:,2);
% for time t = 0.1
U_vis_2_first_good = reshape(numer_sol_first_good(:,2),n_x-1,[])';
U_vis_2_first_good = [zeros(n_x-1,1),U_vis_2_first_good,zeros(n_x-1,1)];
U_vis_2_first_good = [zeros(1,n_y+1);U_vis_2_first_good;zeros(1,n_y+1)];
error_matrix_2_first_good = abs(U_exact(:,:,2,2) - U_vis_2_first_good);
max_error_2_first_good = max(error_matrix_2_first_good,[],'all');

% for time t = 0.5
U_vis_6_first_good = reshape(numer_sol_first_good(:,6),n_x-1,[])';
U_vis_6_first_good = [zeros(n_x-1,1),U_vis_6_first_good,zeros(n_x-1,1)];
U_vis_6_first_good = [zeros(1,n_y+1);U_vis_6_first_good;zeros(1,n_y+1)];
error_matrix_6_first_good = abs(U_exact(:,:,2,6) - U_vis_6_first_good);
max_error_6_first_good = max(error_matrix_6_first_good,[],'all');

% for time t = 1
U_vis_11_first_good = reshape(numer_sol_first_good(:,11),n_x-1,[])';
U_vis_11_first_good = [zeros(n_x-1,1),U_vis_11_first_good,zeros(n_x-1,1)];
U_vis_11_first_good = [zeros(1,n_y+1);U_vis_11_first_good;zeros(1,n_y+1)];
error_matrix_11_first_good = abs(U_exact(:,:,2,11) - U_vis_11_first_good);
max_error_11_first_good = max(error_matrix_11_first_good,[],'all');

%% For a = 1
numer_sol_second_good = numerical_solution(:,:,3);
% for time t = 0.1
U_vis_2_second_good = reshape(numer_sol_second_good(:,2),n_x-1,[])';
U_vis_2_second_good = [zeros(n_x-1,1),U_vis_2_second_good,zeros(n_x-1,1)];
U_vis_2_second_good = [zeros(1,n_y+1);U_vis_2_second_good;zeros(1,n_y+1)];
error_matrix_2_second_good = abs(U_exact(:,:,3,2) - U_vis_2_second_good);
max_error_2_second_good = max(error_matrix_2_second_good,[],'all');

% for time t = 0.5
U_vis_6_second_good = reshape(numer_sol_second_good(:,6),n_x-1,[])';
U_vis_6_second_good = [zeros(n_x-1,1),U_vis_6_second_good,zeros(n_x-1,1)];
U_vis_6_second_good = [zeros(1,n_y+1);U_vis_6_second_good;zeros(1,n_y+1)];
error_matrix_6_second_good = abs(U_exact(:,:,3,6) - U_vis_6_second_good);
max_error_6_second_good = max(error_matrix_6_second_good,[],'all');

% for time t = 1
U_vis_11_second_good = reshape(numer_sol_second_good(:,11),n_x-1,[])';
U_vis_11_second_good = [zeros(n_x-1,1),U_vis_11_second_good,zeros(n_x-1,1)];
U_vis_11_second_good = [zeros(1,n_y+1);U_vis_11_second_good;zeros(1,n_y+1)];
error_matrix_11_second_good = abs(U_exact(:,:,3,11) - U_vis_11_second_good);
max_error_11_second_good = max(error_matrix_11_second_good,[],'all');

%% For a = 5
numer_sol_third_good = numerical_solution(:,:,4);
% for time t = 0.1
U_vis_2_third_good = reshape(numer_sol_third_good(:,2),n_x-1,[])';
U_vis_2_third_good = [zeros(n_x-1,1),U_vis_2_third_good,zeros(n_x-1,1)];
U_vis_2_third_good = [zeros(1,n_y+1);U_vis_2_third_good;zeros(1,n_y+1)];
error_matrix_2_third_good = abs(U_exact(:,:,4,2) - U_vis_2_third_good);
max_error_2_third_good = max(error_matrix_2_third_good,[],'all');

% for time t = 0.5
U_vis_6_third_good = reshape(numer_sol_third_good(:,6),n_x-1,[])';
U_vis_6_third_good = [zeros(n_x-1,1),U_vis_6_third_good,zeros(n_x-1,1)];
U_vis_6_third_good = [zeros(1,n_y+1);U_vis_6_third_good;zeros(1,n_y+1)];
error_matrix_6_third_good = abs(U_exact(:,:,4,6) - U_vis_6_third_good);
max_error_6_third_good = max(error_matrix_6_third_good,[],'all');

% for time t = 1
U_vis_11_third_good = reshape(numer_sol_third_good(:,11),n_x-1,[])';
U_vis_11_third_good = [zeros(n_x-1,1),U_vis_11_third_good,zeros(n_x-1,1)];
U_vis_11_third_good = [zeros(1,n_y+1);U_vis_11_third_good;zeros(1,n_y+1)];
error_matrix_11_third_good = abs(U_exact(:,:,4,11) - U_vis_11_third_good);
max_error_11_third_good = max(error_matrix_11_third_good,[],'all');