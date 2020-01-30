function data = fftshift2(data)
    % Like fftshift, but operates only on the first two dimensions
    % (even in the case of a stack of frames)
    %  - Damien Loterie (02/2014)

    % Get the size of the data
    y = size(data, 1);
    x = size(data, 2);
    
    % Find the length of the positive frequency axis
    yp = ceil(y/2);
    xp = ceil(x/2);
    
    % Remap
    data(1:y, 1:x, :) = data([(yp+1):y 1:yp], [(xp+1):x 1:xp], :);
 
end

