% IMage CLOUD generator
%   To visualize a point cloud in 2D where each point corresponds to an image,
%  this function generates a scatter plot with example images rendered over the 
%  points.
% Input:
%   H - 2-by-N matrix with the location for each observation in the plane.
%   I - dims(1)*dims(2)-by-N matrix with image data for each point in the plane.
%   dims - 2D vector with the height and width of each image. Reshaping the columns of
%   	I to this size should produce each image.
% Output:
%   None (creates a figure).

function [] = imcloud(H, I, dims)
    if size(H,2) == 2 && size(H,1) > 2
        H = H';
    end
    if dims(1) * dims(2) ~= size(I,1)
        error('Columns of I must have the same number of elements as given by dims.');
    end
    
    SUBIMG_PC1_HALF_SIZE = (max(H(1,:)) - min(H(1,:)))/50;
    SUBIMG_PC2_HALF_SIZE = (max(H(2,:)) - min(H(2,:)))/50;
    SUBIMG_SPACING = 2*SUBIMG_PC1_HALF_SIZE;

    plot(H(1,:),H(2,:),'.k');
    title('Cloud','FontSize',22);
    xlabel('X','FontSize',22);
    ylabel('Y','FontSize',22);
    hold on;
    DONE = [1e80,1e80];
    for i = 1:size(I,2)
        CENTER = [H(1,i),H(2,i)];
        if sqrt(min(sum((repmat(CENTER,[size(DONE,1),1]) - DONE).^2,2))) > SUBIMG_SPACING
            imagesc(...
                CENTER(1) + SUBIMG_PC1_HALF_SIZE*[-1,1],...
                CENTER(2) + SUBIMG_PC2_HALF_SIZE*[-1,1],...
                flipud(reshape(I(:,i),dims)),[0,max(max(I(:,i)))]);
            DONE = [DONE;CENTER];
        end
    end
    colormap gray;
    hold off;
    set(gcf,'Color','w');
end