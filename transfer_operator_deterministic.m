function [eigenvectors,eigenvalues,singular_values,reduced_basis,reduced_dimension,tol_val,energy] = transfer_operator_deterministic(mass,stiffness,rhs_matrix,hi_fi_solution,time_grid,nt_loc,kt_loc,tol)
    
    % setting integers for applying implicit euler!
    ints = [31;61;91;121;151;181;211;241;271;300];
    intsz = size(ints,1);
    startpoints = zeros(intsz,1);
    endpoints = zeros(intsz,1);
    aux = nt_loc - kt_loc + 1;
    u_0 = zeros(size(mass,2),intsz);
    rhs_for_loop = zeros(size(rhs_matrix,1),nt_loc+1,intsz);
    for j=1:intsz % this loop creates the starting & ending time points to plug into the implicit Euler function
        startpoints(j) = time_grid(ints(j)-nt_loc);
        endpoints(j) = time_grid(ints(j));
        u_0(:,j) = hi_fi_solution(:,ints(j));
        rhs_for_loop(:,:,j) = rhs_matrix(:,ints(j)-nt_loc:ints(j));
    end
    
    % for the solutions that will be kept
    
    solutions_temp = zeros(size(mass,2),nt_loc+1,intsz);
    solutions = zeros(size(mass,2),intsz*aux+nt_loc);

    % solving the PDE locally and storing the last aux # of solutions for
    % reduced basis generation!
    parfor (j=1:intsz,8)
        solutions_temp(:,:,j) = euler_method(u_0(:,j),startpoints(j),endpoints(j),nt_loc,mass,stiffness,rhs_for_loop(:,:,j));
    end

    for j=1:intsz
        solutions(:,(j-1)*aux+1:j*aux+1) = solutions_temp(:,kt_loc:end,j);
    end

    % adding in initial conditions for the last several (just to keep it
    % consistent with the random algorithm)
    ht = time_grid(2) - time_grid(1);
    initial_temp = euler_method(hi_fi_solution(:,1),time_grid(1),time_grid(1)+nt_loc*ht,nt_loc,mass,stiffness,rhs_matrix);
    solutions(:,end:-1:end-nt_loc) = initial_temp(:,1:nt_loc+1);

    % solving the generalized eigenproblem to find the reduced basis
    matrix = (solutions')*(mass)*(solutions);
    matrix = 0.5.*(matrix + matrix');
    [eigenvectors, eigenvalues] = eig(matrix);

    % determine the dimension of the reduced space
    [eigenvalues,I] = sort(diag(eigenvalues),'descend'); % 'I' provides information about how the elements were resorted
    eigenvectors = eigenvectors(:,I);
    singular_values = real(sqrt(eigenvalues));
    [n,~] = size(singular_values);
    tol_val = zeros(1,n);
    energy = zeros(1,n);
    for i = 1:n
        C = sqrt(sum(singular_values(i+1:end).^2))/sqrt(sum(singular_values.^2));
        E = sqrt(sum(singular_values(1:i).^2))/sqrt(sum(singular_values.^2));
        X = ['The C value at run ',num2str(i),' is: ',num2str(C), '; the percentage of the energy accounted for is: ',num2str(100*E,16),'%'];
        disp(X);
        tol_val(i) = C;
        energy(i) = E;
        if C <= tol
            reduced_dimension = i+1;
            break
        end
    end

    tol_val = tol_val(1:reduced_dimension);
    energy = energy(1:reduced_dimension);
    energy = 100.*energy;
    % compute the reduced basis
    reduced_basis = solutions*eigenvectors(:,1:reduced_dimension);
    for i = 1:reduced_dimension
        reduced_basis(:,i) = reduced_basis(:,i)/singular_values(i); % scales reduced basis by corresponding singular value
    end
    
    % need to re-orthonormalize

    reduced_basis = gs_ortho(reduced_basis,mass);
end