function [rand_red_basis,rand_solutions,rand_solutions_temp,initial_temp,rand_u_0] = rand_red_basis_generation(n_rand,n_time_steps,kt,prob,tol,time_grid,u_0,mass,stiffness,rhs_matrix)

% inputs are:
% n_rand - number of randomly chosen time points
% n_time_steps - number of time steps for local PDE simulations
% kt - starting time step for collecting snapshots (require kt less than or
    % equal to n_time_steps)
% prob - probability distribution
% tol - tolerance
% time_grid - includes all the time points in the global problem

% right now, we're just going to do uniform distribution, since I'm not
% sure how MATLAB does arbitrary distributions!

% generating n_rand random integers from {0,...,number of time points}
t_sz = size(time_grid,1);
rand_ints = randi([0 t_sz],n_rand,1); % right now, this is just uniform

% making sure we can step back in time
rand_ints = rand_ints(rand_ints > n_time_steps);
[sz_rand,~] = size(rand_ints);

startpoints = zeros(sz_rand,1);
endpoints = zeros(sz_rand,1);
for j = 1:sz_rand
    startpoints(j) = time_grid(rand_ints(j) - n_time_steps);
    endpoints(j) = time_grid(rand_ints(j));
end
% disp(startpoints);
% disp(endpoints); 

rand_u_0 = randn(size(mass,2),sz_rand);
% disp(rand_u_0);
for j = 1:sz_rand
    rand_u_0(:,j) = stiffness*rand_u_0(:,j);
    % the stiffness matrix will need to be changed when it's time-dependent
end


aux = n_time_steps-kt+1;
rand_solutions_temp = zeros(size(mass,2),n_time_steps+1,sz_rand);
rand_solutions = zeros(size(mass,2),sz_rand*aux+n_time_steps);

parfor (j = 1:sz_rand,8)
    start_temp = startpoints(j);
    end_temp = endpoints(j);
    rhs_temp = rhs_matrix(:,rand_ints(j)-n_time_steps:rand_ints(j)+1);
    rand_solutions_temp(:,:,j) = euler_method(rand_u_0(:,j),start_temp,end_temp,n_time_steps,mass,stiffness,rhs_temp);
    % NOTE: need to truncate rhs_matrix for this step based on the random
    % integers - I will figure this out once I'm feeling more coherent
    % we can keep this out for now but go back and add it once we see what
    % the other stuff is doing
end

for j = 1:sz_rand
    rand_solutions(:,(j-1)*aux+1:j*aux+1) = rand_solutions_temp(:,kt:end,j);
end

% puts in initial data for the remaining columns of the random solution
% matrix
ht = time_grid(2) - time_grid(1);
% disp(ht);
initial_temp = euler_method(u_0,time_grid(1),time_grid(1) + n_time_steps*ht,n_time_steps,mass,stiffness,rhs_matrix);
rand_solutions(:,end:-1:end-n_time_steps) = initial_temp(:,1:n_time_steps+1);

% computing the SVD of the random solution matrix rand_solutions
[rand_U,rand_singular_vals,~] = svd(rand_solutions);
rand_singular_vals = diag(rand_singular_vals);
    for j = 1:size(rand_singular_vals,1)
        C = sqrt(sum(rand_singular_vals(j+1:end).^2))/sqrt(sum(rand_singular_vals.^2));
        if C <= tol
            red_dim = j+1;
            break
        end
    end

rand_red_basis = rand_U(:,1:red_dim);
end
