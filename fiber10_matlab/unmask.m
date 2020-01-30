function frames = unmask(data, mask)
    % frames = unmask(data, mask)
    % Reconstruct 2D frames from 1D data, using the given mask.
    % This function can also work on multiple frames. In this case, either 
    % each column of data represents a different frame, or data is a single
    % vector containing all the frames one after another.
    %  - Damien Loterie (02/2014)
    %    Updated for N-dimensional frames (05/2016)
    
    % Fool proofing
    dims_data = size(data);
    dims_mask = size(mask);
    N_mask = sum(mask(:));
    N_stack = dims_data(2);
    if numel(dims_data)~=2
        error('The input dataset is expect to be 2-dimensional.');
    end
    if dims_data(1)~=N_mask
        error('The mask is incompatible with the size of the data columns.');
    end
    if numel(dims_mask)>2
        N_dims = numel(dims_mask);
    elseif dims_mask(2)>1
        N_dims = 2;
    else
        N_dims = 1;
    end

    % Initialize output frame stack
    frames = zeros([dims_mask(1:N_dims) N_stack], 'like', data);
    
    % Copy data
    frames(repmat(mask, [ones(1, N_dims) N_stack])) = data;    
end