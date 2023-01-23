clear all

% for defining solution % another test case to work on is the stove problem 

f_x = cell(1,1);

f_x{1,1} = @(x) 4*pi*sin(2*pi*x);

f_y = cell(1,1);

f_y{1,1} = @(y) 2*pi*sin(2*pi*y);

f_t = cell(1,1);

f_t{1,1} = @(t) exp(-t);

% Choose number of elements in x and y direction 

n_x = 500;
n_y = 500;

% Assemble matrices (with _bc means boundary conditions):

[A_bc,M_bc,F,grid_x,grid_y]=FE_assembly(n_x,n_y,f_x,f_y);

% Visualization for some time step t_j

% Exact solution

u = @(x,y) sin(2*pi*x).*sin(2*pi*y); 

[X,Y] = meshgrid(grid_x,grid_y);
U_exact_in_space = u(X,Y);

t = 1;
U_exact_at_time_t = exp(-t).*U_exact_in_space; % NOTE: t needs to be specified

figure(1) 
surf(X,Y,U_exact_at_time_t); shading interp

% Numerical solution of \partial_t u -\Delta u = f

% inputs for the Euler method
T_start = 0;
T_end = 1;
time_steps = 10;
step_size = (T_end - T_start)/time_steps;

% First assemble u_0:
u_0 = u(X,Y);
u_0 = u_0(:,2:end-1);
u_0 = u_0(2:end-1,:);
u_0 = reshape(u_0,[],1);

% evaluating the euler method for different time steps
rhs_matrix = zeros(size(F,1),time_steps+1);

for i = 1:time_steps+1
    rhs_matrix(:,i) = (1 - 1/(8*pi*pi)).*exp(-(step_size)*(i-1)).*F;
end

%% Numerical solution & error for 10 time steps
numerical_solution_10 = euler_method(u_0,T_start,T_end,time_steps,M_bc,A_bc,rhs_matrix);

U_vis_10 = reshape(numerical_solution_10(:,end),n_x-1,[])';

% Now add the zero boundary conditions:

U_vis_10 = [zeros(n_x -1,1),U_vis_10,zeros(n_x -1,1)]; % add zeros on the left and right side (x=0 and x=1)
U_vis_10 = [zeros(1,n_y+1);U_vis_10;zeros(1,n_y+1)]; % add zeros at the top and bottom (y=0 and y=1)

% for error analysis
error_matrix_10 = abs(U_exact_at_time_t - U_vis_10);
max_error_10 = max(error_matrix_10,[],'all');
    
%% Numerical solution & error for 20 time steps
time_steps = 20;
step_size = (T_end - T_start)/time_steps;

rhs_matrix = zeros(size(F,1),time_steps+1);

for i = 1:time_steps+1
    rhs_matrix(:,i) = (1 - 1/(8*pi*pi)).*exp(-(step_size)*(i-1)).*F;
end

numerical_solution_20 = euler_method(u_0,T_start,T_end,time_steps,M_bc,A_bc,rhs_matrix);

U_vis_20 = reshape(numerical_solution_20(:,end),n_x-1,[])';

% Now add the zero boundary conditions:

U_vis_20 = [zeros(n_x -1,1),U_vis_20,zeros(n_x -1,1)]; % add zeros on the left and right side (x=0 and x=1)
U_vis_20 = [zeros(1,n_y+1);U_vis_20;zeros(1,n_y+1)]; % add zeros at the top and bottom (y=0 and y=1)

% for error analysis
error_matrix_20 = abs(U_exact_at_time_t - U_vis_20);
max_error_20 = max(error_matrix_20,[],'all');

%% Numerical solution & error for 40 time steps
time_steps = 40;
step_size = (T_end - T_start)/time_steps;

rhs_matrix = zeros(size(F,1),time_steps+1);

for i = 1:time_steps+1
    rhs_matrix(:,i) = (1 - 1/(8*pi*pi)).*exp(-(step_size)*(i-1)).*F;
end

numerical_solution_40 = euler_method(u_0,T_start,T_end,time_steps,M_bc,A_bc,rhs_matrix);

U_vis_40 = reshape(numerical_solution_40(:,end),n_x-1,[])';

% Now add the zero boundary conditions:

U_vis_40 = [zeros(n_x -1,1),U_vis_40,zeros(n_x -1,1)]; % add zeros on the left and right side (x=0 and x=1)
U_vis_40 = [zeros(1,n_y+1);U_vis_40;zeros(1,n_y+1)]; % add zeros at the top and bottom (y=0 and y=1)

% for error analysis
error_matrix_40 = abs(U_exact_at_time_t - U_vis_40);
max_error_40 = max(error_matrix_40,[],'all');

%% For 80 time steps

time_steps = 80;
step_size = (T_end - T_start)/time_steps;

rhs_matrix = zeros(size(F,1),time_steps+1);

for i = 1:time_steps+1
    rhs_matrix(:,i) = (1 - 1/(8*pi*pi)).*exp(-(step_size)*(i-1)).*F;
end

numerical_solution_80 = euler_method(u_0,T_start,T_end,time_steps,M_bc,A_bc,rhs_matrix);

U_vis_80 = reshape(numerical_solution_80(:,end),n_x-1,[])';

% Now add the zero boundary conditions:

U_vis_80 = [zeros(n_x -1,1),U_vis_80,zeros(n_x -1,1)]; % add zeros on the left and right side (x=0 and x=1)
U_vis_80 = [zeros(1,n_y+1);U_vis_80;zeros(1,n_y+1)]; % add zeros at the top and bottom (y=0 and y=1)

% for error analysis
error_matrix_80 = abs(U_exact_at_time_t - U_vis_80);
max_error_80 = max(error_matrix_80,[],'all');

%% The plot for error analysis

hold off 
xdata = [10 20 40 80];
ydata = [max_error_10, max_error_20, max_error_40, max_error_80];
figure(3)
p = semilogy(xdata,ydata);
grid on
p.LineStyle = ":";
p.Color = "r";
p.Marker = ".";
p.LineWidth = 1;
p.MarkerSize = 12;
hold on
ydatacomp = [max_error_10, max_error_10/2, max_error_10/4, max_error_10/8];
q = semilogy(xdata,ydatacomp);
q.LineStyle = "--";
q.Color = "b";
q.Marker = "o";
q.LineWidth = 1;
q.MarkerSize = 6;
legend('Empirical error','Expected error based on error at 10 time steps');
xlabel('Number of time steps','Interpreter','latex');
ylabel('Maximum error at $t=1$','Interpreter','latex');