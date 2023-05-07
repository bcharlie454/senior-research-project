%% Description

% In this file, we will specifically test the case of discontinuous stoves
% being turned on one after the other -- i.e., one stove on a certain
% subdomain will be on, then switch off as a second stove on a different
% subdomain is switched on. 

% We will create reduced bases for subsets of the time interval (i.e., only
% taking snapshots from certain subintervals of I = [t0,t1]) as well as
% create a reduced basis for the whole time interval.

% Then, we will see if unions of the bases overthe subintervals produce a 
% model in comparison with the whole-interval model (by analyzing error).

%% Source terms, time function, other inputs

% the spatial domain we'll be considering is D = [0,1] x [0,1]
% the time domain will be I = [0,2]

% conductivity and advection coefficients
a = 0.01;
b = [0.3 0];

x0 = 0;
x1 = 1;
y0 = 0;
y1 = 1;

n_x = 100;
n_y = 100;

h_x = 1/n_x;
h_y = 1/n_y;

x_grid = [x0:h_x:x1]';
y_grid = [y0:h_y:y1]';
[X,Y] = meshgrid(x_grid,y_grid);

t0 = 0;
t1 = 2;
h_t = 1/100;
t_grid = [t0:h_t:t1]';
time_steps = (t1-t0)/h_t;

% spatial source terms: stove 1 will be located on the subdomain D_1 =
% [0.1,0.2] x [0.1,0.2] and stove 2 on D_2 = [0.3,0.4] x [0.3,0.4]
f_x = cell(2,1);
f_x{1,1} = @(x) (x>=0.1).*(x<=0.2);
f_x{2,1} = @(x) (x>=0.3).*(x<=0.4);

f_y = cell(2,1);
f_y{1,1} = @(y) (y>=0.1).*(y<=0.2);
f_y{2,1} = @(y) (y>=0.3).*(y<=0.4);

% the stoves will be turned on in a stepwise fashion
f_t = cell(2,1);
f_t{1,1} = @(t) 25.*(t>=0.5).*(t<1.0); % so that stove 1 will turn off at t = 1.0
f_t{2,1} = @(t) 25.*(t>=1.0).*(t<=1.5); % so that stove 2 will turn on at t = 1.0

% initial conditions
u_0 = @(x,y) sin(pi*x).*sin(pi*y) + sin(2*pi*x).*sin(2*pi*y) + sin(3*pi*x).*sin(3*pi*y);
u_0_vis = u_0(X,Y);
u_0_vis = u_0_vis(:,2:end-1);
u_0_vis = u_0_vis(2:end-1,:);
u_0_vis = u_0_vis(:);

% we will also be assuming homogeneous Dirichlet BCs

%% Producing the FE space and high-fidelity solution

% producing the FE space based on the spatial source terms

[stiffs_matrix,mass_matrix,F,~,~] = FE_adv_diff_assemble(a,b,f_x,f_y,n_x,n_y,x0,x1,y0,y1);

[F1,F2] = size(F);

rhs_matrix = zeros(F1,F2,time_steps+1); % for the Euler method

% creates the RHS matrix corresponding to the source terms f = sum_{i=1}^4
% f_t{i,1}f_x{i,1}f_y{i,1}

one_time = f_t{1,1}(t_grid);
two_time = f_t{2,1}(t_grid);

for k = 1:time_steps+1
    rhs_matrix(:,1,k) = F(:,1).*one_time(k);
    rhs_matrix(:,2,k) = F(:,2).*two_time(k);
end

rhs_matrix = sum(rhs_matrix,2);
rhs_matrix = reshape(rhs_matrix,F1,time_steps+1);

% computes the high-fidelity solution:

hf_solution = euler_method(u_0_vis,t0,t1,time_steps,mass_matrix,stiffs_matrix,rhs_matrix);

%% Constructing a reduced basis for the whole time interval

% in this section, we will start with a "proof-of-concept" approach where
% we use the function transfer_operator_deterministic to obtain a reduced 
% basis based on specific snapshots which we know are important 
% (since we know the source terms)

% the variables in this section will typically be labeled "_whole" to
% indicate that the reduced basis is constructed by using information from
% the whole time interval

ints_whole = [16;31;51;76;96;102;116;136;152;176]; % may need to pick these a little more strategically but for now let's see

nt_loc = 15;
kt_loc = 13;
tol_transfer = 10^(-8);

% constructing the whole reduced basis
[eigvecs_whole,eigvals_whole,sing_vals_whole,red_bas_whole,red_dim_whole,tol_val_whole,energy_whole] = transfer_operator_deterministic(mass_matrix,stiffs_matrix,rhs_matrix,hf_solution,t_grid,ints_whole,nt_loc,kt_loc,tol_transfer);

% creating reduced mass, stiffness & RHS matrices, along with reduced
% initial conditions to plug into the Euler method algorithm
red_mass_whole = red_bas_whole'*mass_matrix*red_bas_whole;
red_stiffs_whole = red_bas_whole'*stiffs_matrix*red_bas_whole;
red_rhs_whole = red_bas_whole'*rhs_matrix;
red_u_0_whole = ((red_mass_whole)^(-1))*red_bas_whole'*mass_matrix*u_0_vis; % remember that this needs to be changed if we're starting at a time point different than t = 0

% calculating the reduced solution
red_sol_whole = euler_method(red_u_0_whole,t0,t1,time_steps,red_mass_whole,red_stiffs_whole,red_rhs_whole);

[sz_red_whole_x,sz_red_whole_y] = size(red_sol_whole); % good to store this information for future steps in code

% generates reduced solution in FE space (to be used for error analysis)
red_sol_whole_for_error = red_bas_whole*red_sol_whole;

% error analysis for the whole reduced solution:



%% Reduced basis for the first stove

% here we'll consider a reduced basis with data from the time subinterval 
% I_1 = [0,1.1] - we're going to have a little bit of overlap between this
% subinterval and the subinterval used for the reduced basis at a later
% time

% we'll use subscripts "_1" for naming variables in this section - the
% procedure though will be the exact same as that in the previous section

ints_1 = [16;31;53;61;71;81;91;101;103;111]; % for snapshot generation
% we'll use the same nt_loc, kt_loc, and tol_transfer as the above

% truncating the RHS & hf_solution matrices so that we are only considering
% the diff eq in the subinterval I_1; we'll also truncate the time grid

hf_sol_trunc_1 = hf_solution(:,1:111);
rhs_trunc_1 = rhs_matrix(:,1:111);

time_subint_1 = t_grid(1:111);

% constructing the reduced basis
[eigvecs_1,eigvals_1,sing_vals_1,red_bas_1,red_dim_1,tol_val_1,energy_1] = transfer_operator_deterministic(mass_matrix,stiffs_matrix,rhs_trunc_1,hf_sol_trunc_1,time_subint_1,ints_1,nt_loc,kt_loc,tol_transfer);

red_mass_1 = red_bas_1'*mass_matrix*red_bas_1;
red_stiffs_1 = red_bas_1'*stiffs_matrix*red_bas_1;
red_rhs_1 = red_bas_1'*rhs_trunc_1;
red_u_0_1 = ((red_mass_1)^(-1))*red_bas_1'*mass_matrix*u_0_vis;

% new time interval
t0_1 = time_subint_1(1);
t1_1 = time_subint_1(end);
t_steps_1 = (t1_1 - t0_1)/h_t;

red_sol_1 = euler_method(red_u_0_1,t0_1,t1_1,t_steps_1,red_mass_1,red_stiffs_1,red_rhs_1);

[sz_red_1_x,sz_red_1_y] = size(red_sol_1);

% reduced solution in FE space:

red_sol_1_for_error = red_bas_1*red_sol_1;

%% Reduced basis for the second stove

% here we'll consider a reduced basis with data from the time subinterval 
% I_2 = [0.9,2] - we're going to have a little bit of overlap between this
% subinterval and the subinterval used for the reduced basis at the earlier
% time (constructed in the previous section)

% we'll use subscripts "_2" for naming variables in this section - the
% procedure though will be the exact same as that in the previous section

% for the integers, we'll need to calculate them with the subinterval in
% mind (i.e. we're starting at time point 0.9 -> 91st grid point if we're
% considering the whole time interval, but 1st gride point for the
% subinterval!)

ints_2 = [106;109;111;121;131;141;151;153;191;201];
ints_2 = ints_2 - 90; % to account for the shift in start time

% same nt_loc, kt_loc, tol_transfer as above

% truncating hf_solution & rhs_matrix, as well as defining I_2

hf_sol_trunc_2 = hf_solution(:,91:end);
rhs_trunc_2 = rhs_matrix(:,91:end);

time_subint_2 = t_grid(91:end);

% constructing the reduced basis
[eigvecs_2,eigvals_2,sing_vals_2,red_bas_2,red_dim_2,tol_val_2,energy_2] = transfer_operator_deterministic(mass_matrix,stiffs_matrix,rhs_trunc_2,hf_sol_trunc_2,time_subint_2,ints_2,nt_loc,kt_loc,tol_transfer);

red_mass_2 = red_bas_2'*mass_matrix*red_bas_2;
red_stiffs_2 = red_bas_2'*stiffs_matrix*red_bas_2;
red_rhs_2 = red_bas_2'*rhs_trunc_2;
red_u_0_2 = ((red_mass_2)^(-1))*red_bas_2'*mass_matrix*hf_sol_trunc_2(:,1); % making sure we have the right ICs for t0 = 0.9

% other inputs for the euler method
t0_2 = 0.9;
t1_2 = 2;
t_steps_2 = (t1_2 - t0_2)/h_t; % note we've used the same step size h_t throughout

% euler method
red_sol_2 = euler_method(red_u_0_2,t0_2,t1_2,t_steps_2,red_mass_2,red_stiffs_2,red_rhs_2);

[sz_red_2_x,sz_red_2_y] = size(red_sol_2);

% generates reduced solution in FE space
red_sol_2_for_error = red_bas_2*red_sol_2;

%% Reduced basis comprised of the union of the two subinterval bases

% Here we will combine the reduced bases red_bas_1 and red_bas_2 and go
% through the same steps as above to generate a reduced solution over the
% whole time interval. This will be to compare the accuracy of the union
% with the accuracy of each basis separately (let's see if it works!)

% we won't need the ints this time - we already have our reduced bases!
% We'll use "_union" to denote the union of the two reduced bases and
% subsequent matrices related to this
% 
% red_bas_union = [red_bas_1,red_bas_2];
% 
% % here we can use the full RHS and hf_solution matrices from before
% 
% red_mass_union = red_bas_union'*mass_matrix*red_bas_union;
% red_stiffs_union = red_bas_union'*stiffs_matrix*red_bas_union;
% red_rhs_union = red_bas_union'*rhs_matrix;
% red_u_0_union = (red_mass_union^(-1))*red_bas_union'*mass_matrix*u_0_vis;
% 
% % remember from before, t0 = 0, t1 = 2, time_steps = 2/(1/100) = 200
% 
% red_sol_union = euler_method(red_u_0_union,t0,t1,time_steps,red_mass_union,red_stiffs_union,red_rhs_union);
% 
% [sz_red_union_x,sz_red_union_y] = size(red_sol_union);
% 
% % solution in FE space
% red_sol_union_for_error = red_bas_union*red_sol_union;

%% Switching from one reduced basis to the next!

% here we will solve the reduced model on t in [0,1] with the first reduced
% basis, and then we will switch to the second reduced basis on t in [1,2]

% creating relevant time points and step numbers
t0_comb = 0;
t1_comb = 1;
t2_comb = 2;

t_steps_comb_1 = (t1_comb - t0_comb)/h_t;
t_steps_comb_2 = (t2_comb - t1_comb)/h_t;

% the reduced solution on t in [0,1) with the first reduced basis
red_sol_comb_temp_1 = euler_method(red_u_0_1,t0_comb,t1_comb,t_steps_comb_1,red_mass_1,red_stiffs_1,red_rhs_1);

% the reduced solution on t in [1,2] with the second reduced basis, using
% the previous reduced solution at the final time step as the initial
% condition

red_sol_comb_temp_2 = euler_method(red_sol_2(:,11),t1_comb,t2_comb,t_steps_comb_2,red_mass_2,red_stiffs_2,red_rhs_2(:,11:end));

% combining the two temp matrices into one solution (appropriately
% truncating so that there's no double counting of specific time steps)
% red_sol_comb = [red_sol_comb_temp_1(:,1:end-1),red_sol_comb_temp_2];

% [sz_red_comb_x,sz_red_comb_y] = size(red_sol_comb);

% we also have to be careful about how we do the computation for the error
% (since we're in different bases depending on the interval!!)

red_sol_comb_for_error_temp_1 = red_bas_1*red_sol_comb_temp_1(:,1:end-1);
red_sol_comb_for_error_temp_2 = red_bas_2*red_sol_comb_temp_2;

red_sol_comb_for_error = [red_sol_comb_for_error_temp_1,red_sol_comb_for_error_temp_2];

%% Error analysis for the subintervals

% for the reduced basis of the whole interval and the combined reduced basis

[~,sz_red_comb_y] = size(red_sol_comb_for_error);

H1_errors_over_time_whole = zeros(1,sz_red_whole_y);
H1_norms_FE_space_whole = zeros(1,sz_red_whole_y);
H1_errors_over_time_relative_whole = zeros(1,sz_red_whole_y);

H1_errors_over_time_comb = zeros(1,sz_red_comb_y);
H1_norms_FE_space_comb = zeros(1,sz_red_comb_y);
H1_errors_over_time_relative_comb = zeros(1,sz_red_comb_y);

for i = 1:sz_red_whole_y
    error_temp_whole = hf_solution(:,i) - red_sol_whole_for_error(:,i);
    error_temp_comb = hf_solution(:,i) - red_sol_comb_for_error(:,i);
    H1_errors_over_time_whole(i) = sqrt(error_temp_whole'*stiffs_matrix*error_temp_whole);
    H1_errors_over_time_comb(i) = sqrt(error_temp_comb'*stiffs_matrix*error_temp_comb);
    H1_norms_FE_space_whole(i) = sqrt(hf_solution(:,i)'*stiffs_matrix*hf_solution(:,i));
    H1_norms_FE_space_comb(i) = H1_norms_FE_space_whole(i);
    H1_errors_over_time_relative_whole(i) = H1_errors_over_time_whole(i)/H1_norms_FE_space_whole(i);
    H1_errors_over_time_relative_comb(i) = H1_errors_over_time_comb(i)/H1_norms_FE_space_comb(i);
end

% for the subintervals
H1_errors_over_time_1 = zeros(1,sz_red_1_y);
H1_errors_over_time_2 = zeros(1,sz_red_2_y);

H1_norms_FE_space_1 = zeros(1,sz_red_1_y);
H1_norms_FE_space_2 = zeros(1,sz_red_2_y);

H1_errors_over_time_relative_1 = zeros(1,sz_red_1_y);
H1_errors_over_time_relative_2 = zeros(1,sz_red_2_y);

for i = 1:sz_red_1_y
    error_temp_1 = hf_sol_trunc_1(:,i) - red_sol_1_for_error(:,i);
    error_temp_2 = hf_sol_trunc_2(:,i) - red_sol_2_for_error(:,i);
    H1_errors_over_time_1(i) = sqrt(error_temp_1'*stiffs_matrix*error_temp_1);
    H1_errors_over_time_2(i) = sqrt(error_temp_2'*stiffs_matrix*error_temp_2);
    H1_norms_FE_space_1(i) = sqrt(hf_sol_trunc_1(:,i)'*stiffs_matrix*hf_sol_trunc_1(:,i));
    H1_norms_FE_space_2(i) = sqrt(hf_sol_trunc_2(:,i)'*stiffs_matrix*hf_sol_trunc_2(:,i));
    H1_errors_over_time_relative_1(i) = H1_errors_over_time_1(i)/H1_norms_FE_space_1(i);
    H1_errors_over_time_relative_2(i) = H1_errors_over_time_2(i)/H1_norms_FE_space_2(i);
end

figure(1)
o = semilogy(t_grid,H1_errors_over_time_relative_whole);
grid on
o.LineWidth = 2;
o.LineStyle = "--";
o.Color = 'black';

hold on
p = semilogy(time_subint_1,H1_errors_over_time_relative_1); 
grid on
p.LineWidth = 2;
p.Color = 'r';

hold on
q = semilogy(time_subint_2,H1_errors_over_time_relative_2); 
grid on
q.LineWidth = 2;
q.Color = 'b';

hold on
r = semilogy(t_grid,H1_errors_over_time_relative_comb);
grid on
r.LineWidth = 2.5;
r.Color = [0.4660 0.6740 0.1880];
r.LineStyle = "-.";
hold on
xlabel('$t$','Interpreter','latex','FontSize',12);
ylabel('Relative $H^1$ errors over time','Interpreter','latex','FontSize',12);
legend('Reduced basis over whole interval','Reduced basis over $t\in[0,1.1]$','Reduced basis over $t\in[0.9,2]$','Combined reduced basis','Interpreter','latex','FontSize',12);