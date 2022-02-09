% IMage GRID generator
%   Given a bunch of little images, this generates a single figure which shows
%  a lot of them.
% Input:
%   I - dims(1)*dims(2)-by-N matrix with image data for each of N images.
%   dims - 2D vector with the height and width of each image. Reshaping the columns of
%   	I to this size should produce each image.
%   gridsz - The size of the grid to show (for example, [2,3] will show six images in total
%   	laid out in a grid with two rows).
% Output:
%   None (creates a figure).

function [] = imgrid(I, dims, gridsz)
    if dims(1) * dims(2) ~= size(I,1)
        error('Columns of I must have the same number of elements as given by dims.');
    end
    
    M = zeros(dims(1) * gridsz(1), dims(2) * gridsz(2));
    k = 1;
    for i = 1:gridsz(1)
        for j = 1:gridsz(2)
            is = (1 + (i-1)*dims(1)):((i)*dims(1));
            js = (1 + (j-1)*dims(2)):((j)*dims(2));
            M(is, js) = reshape(I(:,k), dims);
            k = k + 1;
            if k > size(I,2)
                break;
            end
        end
        if k > size(I,2)
        	break;
        end
    end
    imagesc(M);
    colormap gray;
    set(gcf,'Color','w');
	set(gca,'XTickLabel',[]);
	set(gca,'YTickLabel',[]);
end