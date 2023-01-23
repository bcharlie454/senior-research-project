function solution = euler_method(u_0, T_start, T_end, N_steps, mass, stiffness, rhs_matrix)
    
    % Time step size:
    ht = (T_end - T_start)/N_steps;

    % allocating memory for the solution at every time point, incorporating
    % initial conditions:
    solution = zeros(size(mass,2),N_steps+1);
    % this will take the size of the mass matrix along the 2nd axis (need
    % to figure out why this is the case - do some math!)
    solution(:,1) = u_0;

    % the implicit Euler method:
    for i=1:N_steps
        % right hand side at the current time point
        F_temp = rhs_matrix(:,i+1);
        rhs_temp = ht.*F_temp + mass*solution(:,i); % will need to check that the matrix and vector dimensions all match up
        % will be element-wise multiplication for the ht times F_temp, and
        % matrix multiplication for the mass matrix times the solution
        % vector at the (i-1)-th time point
        solution(:,i+1) = (mass + ht.*stiffness)\rhs_temp;
        % the above solves the system:
        % (M + ht*A_{i+1})u_{i+1} = ht*F_{i+1} + mass*u_{i}
    end



end