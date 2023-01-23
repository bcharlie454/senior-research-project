function new_basis = gs_ortho(basis,inner_prod)

    % this function performs the Gram-Schmidt method on a set of vectors
    % which is the `basis' input

    % the method is performed w.r.t. a weighted inner product, described by
    % the `inner_prod' input

    [n,k] = size(basis);
    new_basis = zeros(n,k);
    new_basis(:,1) = basis(:,1)./(sqrt(basis(:,1)'*inner_prod*basis(:,1)));
    for i = 2:k
        new_basis(:,i) = basis(:,i);
        for j = 1:i-1
            new_basis(:,i) = new_basis(:,i) - (new_basis(:,j)'*inner_prod*new_basis(:,i)).*new_basis(:,j);
        end
    new_basis(:,i) = new_basis(:,i)./(sqrt(new_basis(:,i)'*inner_prod*new_basis(:,i)));
    end
end