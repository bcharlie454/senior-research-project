%% Description

% Here we will test the Euler method after applying the advection-diffusion
% FE algorithm to (2+1)-dimensional test case! We'll make sure that we
% obtain good convergence results at each time step

%% Exact Solution in space, setting inputs, etc.

u = @(x,y) sin(2*pi*x).*sin(2*pi*y); % this is the spatial component of the exact solution
a = 1;
b = [1 0]; % if b had a nonzero y-component as well, then we could not separate f(x,y) nicely into f(x)f(y)
% maybe will need to generalize the code at some point to deal with this!

% RHS based on our prescribed solution
f_x = cell(1,1);
f_x{1,1} = @(x) (8*pi*pi - 1)*sin(2*pi*x) + 2*pi*cos(2*pi*x);

f_y = cell(1,1);
f_y{1,1} = @(y) sin(2*pi*y);

x0 = 0;
x1 = 1;
y0 = 0;
y1 = 1;

%% Finite Element assembly

n_x = 100;
n_y = 100;

[A_bc,M_bc,F,x_grid,y_grid] = FE_adv_diff_assemble(a,b,f_x,f_y,n_x,n_y,x0,x1,y0,y1);

% % reshaping the numerical solution for plotting
% num_sol_vis = zeros(n_x-1,n_y-1);
% 
% % this makes, e.g. the first row of num_sol_vis a vector of solutions in
% % the y direction at the first x value (h_x)
% num_sol_vis(:) = numerical_solution;
% 
% % incorporating homogeneous Dirichlet BCs
% 
% num_sol_vis = [zeros(1,n_y-1);num_sol_vis;zeros(1,n_y-1)]; % adds zeros in y direction for x = 0 and x = 1
% num_sol_vis = [zeros(n_x+1,1) num_sol_vis zeros(n_x+1,1)]; % adds zeros in x direction for y = 0 and y = 1

%% Time stepping scheme

t0 = 0;
t1 = 1;
time_grid = [0:1/100:t1]';
time_steps = size(time_grid,1) - 1;

rhs_matrix = zeros(size(F,1),time_steps+1);

for i = 1:time_steps+1
    rhs_matrix(:,i) = exp(-time_grid(i)).*F; % constructing the right-hand side at each time step for the Euler method
end

[X,Y] = meshgrid(x_grid,y_grid);

u_0 = u(X,Y);
u_0 = u_0(2:end-1,:);
u_0 = u_0(:,2:end-1);
u_0 = u_0(:); % this should reshape things in the correct way, if not though we can change it!

num_sol_in_time = euler_method(u_0,t0,t1,time_steps,M_bc,A_bc,rhs_matrix);

error_matrices = zeros(n_x+1,n_y+1,time_steps+1); % this will include BCs just to make sure the reshaping is correct!

%% Error calculations and plotting:

max_exact = zeros(1,time_steps+1);
max_numer = zeros(1,time_steps+1);

for i = 1:time_steps+1
    u_temp_for_max = exp(-time_grid(i)).*u(X,Y);
    max_exact(i) = max(u_temp_for_max,[],"all");
    max_numer(i) = max(num_sol_in_time(:,i),[],"all");
end

figure(1)
p = plot(time_grid,max_exact);
grid on
p.LineStyle = ":";
p.Color = "r";
p.Marker = ".";
p.LineWidth = 1;
p.MarkerSize = 9;
hold on

q = plot(time_grid,max_numer);
q.LineStyle = "--";
q.Color = "b";
q.Marker = "o";
q.LineWidth = 1;
q.MarkerSize = 4.5;

hold on
r = plot(time_grid,exp(-time_grid));
r.LineStyle = "-";
r.Color = [0.4660 0.6740 0.1880];
r.LineWidth = 1.5;

hold on
xlabel('$t$','Interpreter','latex');
ylabel('Maximum value of solution','Interpreter','latex');
legend('Exact solution','Numerical solution','Exponential decay');