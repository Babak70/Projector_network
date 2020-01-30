function data = ifftshift2(data)
    % Like fftshift, but operates only on the first two dimensions
    % (even in the case of a stack of frames)
    %  - Damien Loterie (02/2014)

    % Get the size of the data
    y = size(data, 1);
    x = size(data, 2);
    
    % Find the length of the negative frequency axis
    yn = floor(y/2);
    xn = floor(x/2);
    
    % Remap
    data(1:y, 1:x, :) = data([(yn+1):y 1:yn], [(xn+1):x 1:xn], :);
end

