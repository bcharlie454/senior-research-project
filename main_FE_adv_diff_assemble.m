%% Description:

% This file is to test the function FE_adv_diff_assemble on a 2D spatial
% domain D = [0,1] * [0,1]

% as a first test case, we prescribe the exact solution:
% u(x,y) = sin(2*pi*x) * sin(2*pi*y)

% and see if the FE code approximates it well

%% Exact Solution, setting inputs, etc.

u = @(x,y) sin(2*pi*x).*sin(2*pi*y);
a = 1;
b = [1 0]; % if b had a nonzero y-component as well, then we could not separate f(x,y) nicely into f(x)f(y)
% maybe will need to generalize the code at some point to deal with this!

% RHS based on our prescribed solution
f_x = cell(1,1);
f_x{1,1} = @(x) 4*pi*sin(2*pi*x) + cos(2*pi*x);

f_y = cell(1,1);
f_y{1,1} = @(y) 2*pi*sin(2*pi*y);

%% Finite Element assembly

n_x = 100;
n_y = 100;

[A_bc,M_bc,F,x_grid,y_grid] = FE_adv_diff_assemble(a,b,f_x,f_y,n_x,n_y,0,1,0,1);

numerical_solution = A_bc\F;

% reshaping the numerical solution for plotting
num_sol_vis = zeros(n_x-1,n_y-1);

% this makes, e.g. the first row of num_sol_vis a vector of solutions in
% the y direction at the first x value (h_x)
num_sol_vis(:) = numerical_solution;

% incorporating homogeneous Dirichlet BCs

num_sol_vis = [zeros(1,n_y-1);num_sol_vis;zeros(1,n_y-1)]; % adds zeros in y direction for x = 0 and x = 1
num_sol_vis = [zeros(n_x+1,1) num_sol_vis zeros(n_x+1,1)]; % adds zeros in x direction for y = 0 and y = 1

%% Plotting

% making the grid for plotting
[X,Y] = meshgrid(x_grid,y_grid);

figure(1)
surf(X,Y,num_sol_vis);
shading interp

figure(2)
exact_sol = u(X,Y);
surf(X,Y,exact_sol);
shading interp

%% Error calculations and plotting:

error_matrix = abs(num_sol_vis(2:end-1,2:end-1) - exact_sol(2:end-1,2:end-1));

xdata = x_grid(2:end-1);
ydata = y_grid(2:end-1);

[Xerr,Yerr] = meshgrid(xdata,ydata);

figure(3)
surf(Xerr,Yerr,error_matrix);
shading interp

%% A second test - this time for different a's and b's

a1 = 1/(8*pi*pi);
b1 = [1/(2*pi) 0];

g_x = cell(1,1);
g_x{1,1} = @(x) sin(2*pi*x) + cos(2*pi*x);

g_y = cell(1,1);
g_y{1,1} = @(y) sin(2*pi*y);

%% Finite element assembly, reshaping numerical solution

[A1_bc,M1_bc,F1,x_grid1,y_grid1] = FE_adv_diff_assemble(a1,b1,g_x,g_y,n_x,n_y,0,1,0,1);

numerical_solution_1 = A1_bc\F1;

% for reshaping and plotting
num_sol_1_vis = zeros(n_x-1,n_y-1);
num_sol_1_vis(:) = numerical_solution_1;
num_sol_1_vis = [zeros(1,n_y-1);num_sol_1_vis;zeros(1,n_y-1)];
num_sol_1_vis = [zeros(n_x+1,1) num_sol_1_vis zeros(n_x+1,1)];

%% Plotting the error

[X1,Y1] = meshgrid(x_grid1,y_grid1);
figure(4)
surf(X1,Y1,num_sol_1_vis);
shading interp

figure(5)
surf(X1,Y1,exact_sol);
shading interp

error_matrix_1 = abs(exact_sol(2:end-1,2:end-1) - num_sol_1_vis(2:end-1,2:end-1));

figure(6)
surf(Xerr,Yerr,error_matrix_1);
shading interp

%% Actual (a priori) error analysis

% using previous example as a test case

N = [50,100,200,400];
error_norm_frob = zeros(1,4);
error_2norm = zeros(1,4);

for i=1:4
    n_x_temp = N(i);
    n_y_temp = N(i);
    [A1_bc,~,F1,xgrid_temp,ygrid_temp] = FE_adv_diff_assemble(a1,b1,g_x,g_y,n_x_temp,n_y_temp,0,1,0,1);
    num_sol = A1_bc\F1;
    [X,Y] = meshgrid(xgrid_temp,ygrid_temp);
    ex_sol = u(X,Y);
    ex_sol = ex_sol(2:end-1,:);
    ex_sol = ex_sol(:,2:end-1);
    [sz_x,sz_y] = size(ex_sol);
    error_mat = zeros(sz_x,sz_y);
    error_mat(:) = num_sol;
    error_mat = abs(error_mat - ex_sol);
    error_norm_frob(i) = norm(error_mat,'fro'); % calculates Frobenius norm of the error matrix
    ex_sol = ex_sol';
    ex_sol_vec = ex_sol(:);
    error_vec = abs(num_sol - ex_sol_vec); % for 2-norm error calculation - let's see how this works!
    error_2norm(i) = norm(sqrt(1/(N(i).^2)).*error_vec);
end

% creating the best-fit line based on the data
h1 = 1./(N);
logh1 = log(h1);
logE1 = log(error_2norm);
best_fit_matrix = [ones(1,4); logh1];
best_fit_coeffs = best_fit_matrix'\logE1';
logC1 = best_fit_coeffs(1);
kapp1 = best_fit_coeffs(2);
disp(kapp1);

% plotting the error
figure(7)
p5 = plot(logh1,logE1);
p5.LineStyle = "none";
p5.Marker = '.';
p5.Color = 'r';
p5.LineWidth = 1;
p5.MarkerSize = 12;

hold on
p6 = plot(logh1,logC1+kapp1.*logh1);
p6.LineStyle = "--";
p6.Color = 'b';
p6.LineWidth = 1;

grid on

hold on
legend([p5 p6],'Error Norm','Best-fit Line');
xlabel('$\log(h_x)$','Interpreter','latex');
ylabel('$\log(||E^h||_2)$','Interpreter','latex');