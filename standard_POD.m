function [singular_values, reduced_basis, eigenvectors, eigenvalues, tol_val, reduced_dimension, energy] = standard_POD(mass_matrix, FE_solution, nt_POD, tol_POD)
    % creates the snapshot matrix:
    snapshots = FE_solution(:,1:nt_POD);

    % constructs matrices for eignvalue problem that yields squares of singular values
    matrix = (snapshots')*(mass_matrix)*(snapshots);
    matrix = 0.5.*(matrix + matrix');
    [eigenvectors, eigenvalues] = eigs(matrix,nt_POD);
    % it seems like the eigs function retunrs the k largest eigenvalues,
    % where k = nt_POD
    % also, here the eigenvectors are returned as the first entry
    % the second entry is a diagonal matrix with the eigenvalues as the
    % diagonals

    % determine the dimension of the reduced space
    eigenvalues = diag(eigenvalues);
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
        if C <= tol_POD
            reduced_dimension = i+1;
            break
        end
    end

    tol_val = tol_val(1:reduced_dimension);
    energy = energy(1:reduced_dimension);
    energy = 100.*energy;
    % compute the reduced basis
    reduced_basis = snapshots*eigenvectors(:,1:reduced_dimension);
    for i = 1:reduced_dimension
        reduced_basis(:,i) = reduced_basis(:,i)/singular_values(i); % scales reduced basis by corresponding singular value
        % check this again in text
    end

    % need to re-orthonormalize

    reduced_basis = gs_ortho(reduced_basis,mass_matrix);
end