function data = mask(stack, mask)
    % data = mask(frames, mask)
    % Extract 1D data from N-D images, using the given mask.
    % This function can also work on multiple frames. In this case, the
    % output is a matrix where each column represents one frame.
    %  - Damien Loterie (02/2014)
    %    Updated for N-dimensional frames (05/2016)
    
    % Fool proofing
    dims_stack = size(stack);
    dims_mask = size(mask);
    N_mask = sum(mask(:));
    if numel(dims_mask)==(numel(dims_stack)-1)
        N_dims = numel(dims_mask);
        N_stack = dims_stack(end);
    elseif numel(dims_mask)==numel(dims_stack) && all(dims_stack==dims_mask)
        N_dims = numel(dims_mask);
        N_stack = 1;
    elseif numel(dims_stack)==2 && dims_mask(1)==dims_stack(1) && dims_mask(2)==1
        N_dims = 1;
        N_stack = dims_stack(end);
    else
       error('Wrong frame dimensions.'); 
    end
    if not(all(dims_mask(1:N_dims)==dims_stack(1:N_dims)))
       error('The mask is incompatible with the frames.'); 
    end
       
    % Initialize output
    data = zeros(N_mask, N_stack, 'like', stack);
    
    % Mask the frames
    data(:) = stack(repmat(mask, [ones(1, N_dims) N_stack]));
    
end

