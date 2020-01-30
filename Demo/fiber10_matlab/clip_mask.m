function [mask, ind] = clip_mask(mask)
    % Clip a mask to the smallest rectangular region that contains all the
    % 'true' pixels.
    %  - Damien Loterie (02/2014)

    % Determine clipping region
    my = any(mask,2);
    mx = any(mask,1);
    y1 = find(my,1,'first');
    y2 = find(my,1,'last');
    x1 = find(mx,1,'first');
    x2 = find(mx,1,'last');
    
    % Mask
    mask = mask(y1:y2, x1:x2);
    %mask = mask(any(mask,2),any(mask,1));
    
    % Indices
    if nargout>1
       ind = zeros(2,2);
       ind(1) = y1;
       ind(2) = y2;
       ind(3) = x1;
       ind(4) = x2;
    end
    
end

