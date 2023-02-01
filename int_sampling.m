function [rand_ints,probabilities] = int_sampling(data,time_grid,n_rand,type)
    % the type input specifies what type of sampling is done
    % type should have one of the following values:
    % 0 <-> uniform sampling
    % 1 <-> squared norm sampling
    % 2 <-> rank-3 leverage score sampling
    
    sz_t = size(time_grid,1);
    probabilities = zeros(1,sz_t);
    ints = 1:sz_t;
    frob_norm = norm(data,'fro');    
    
    if type == 0
        rand_ints = randi([0 sz_t],1,n_rand);

    elseif type == 1
        parfor (i = 1:sz_t,4)
            probabilities(i) = norm(data(:,i));
        end
        probabilities = probabilities./frob_norm;
        % this samples n_rand integers from the data with the specified probabilities:
        % it's with replacement
        rand_ints = datasample(ints,n_rand,'Replace',true,'Weights',probabilities);

    elseif type == 2
        [~,~,V] = svds(data,3); % computes 3 right singular vectors of the data matrix
        parfor (i=1:sz_t,4)
            probabilities(i) = (1/3)*sum(V(i,:).^2);
        end
        rand_ints = datasample(ints,n_rand,'Replace',true,'Weights',probabilities);
    end

end