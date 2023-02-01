function [rand_red_basis,rand_eigenvalues] = rand_red_basis_generation(n_rand,n_time_steps,kt,prob_type,tol,time_grid,u_0,mass,stiffness,rhs_matrix)

% inputs are:
% n_rand - number of randomly chosen time points
% n_time_steps - number of time steps for local PDE simulations
% kt - starting time step for collecting snapshots (require kt less than or
    % equal to n_time_steps)
% prob_type - type of probability distribution
    % prob_type == 0 <-> uniform sampling of time points
    % prob_type == 1 <-> squared norm sampling
    % prob_type == 2 <-> rank-3 leverage score sampling
% tol - tolerance
% time_grid - includes all the time points in the global problem

% right now, we're just going to do uniform distribution, since I'm not
% sure how MATLAB does arbitrary distributions!

% generating n_rand random integers from {0,...,number of time points}
rand_ints = int_sampling(rhs_matrix,time_grid,n_rand,prob_type);
% making sure we can step back in time
rand_ints = rand_ints(rand_ints > n_time_steps);
[~,sz_rand] = size(rand_ints);

startpoints = zeros(sz_rand,1);
endpoints = zeros(sz_rand,1);
rhs_for_loop = zeros(size(rhs_matrix,1),n_time_steps+1,sz_rand);
rand_u_0 = randn(size(mass,2),sz_rand);
for j = 1:sz_rand
    startpoints(j) = time_grid(rand_ints(j) - n_time_steps);
    endpoints(j) = time_grid(rand_ints(j));
    rhs_for_loop(:,:,j) = rhs_matrix(:,rand_ints(j)-n_time_steps:rand_ints(j));
    rand_u_0(:,j) = stiffness*rand_u_0(:,j);
end

aux = n_time_steps-kt+1;
rand_solutions_temp = zeros(size(mass,2),n_time_steps+1,sz_rand);
rand_solutions = zeros(size(mass,2),sz_rand*aux+n_time_steps);

parfor (j = 1:sz_rand,8)
    start_temp = startpoints(j);
    end_temp = endpoints(j);
    rand_solutions_temp(:,:,j) = euler_method(rand_u_0(:,j),start_temp,end_temp,n_time_steps,mass,stiffness,rhs_for_loop(:,:,j));
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

% computing the SVD of the general eigenproblem (so we have consistency)
matrix = rand_solutions'*mass*rand_solutions;
matrix = 0.5*(matrix + matrix');
[rand_eigenvectors,rand_eigenvalues] = eig(matrix);
[rand_eigenvalues,I] = sort(diag(rand_eigenvalues),'descend');
rand_eigenvectors = rand_eigenvectors(:,I);
rand_singular_vals = real(sqrt(rand_eigenvalues));
[n,~] = size(rand_singular_vals);
    for j = 1:n
        C = sqrt(sum(rand_singular_vals(j+1:end).^2))/sqrt(sum(rand_singular_vals.^2));
        if C <= tol
            red_dim = j+1;
            break
        end
    end

rand_red_basis = rand_solutions*rand_eigenvectors(:,1:red_dim);
for i = 1:red_dim
    rand_red_basis(:,i) = rand_red_basis(:,i)/rand_singular_vals(i);
end

rand_red_basis = gs_ortho(rand_red_basis,mass); 
end
