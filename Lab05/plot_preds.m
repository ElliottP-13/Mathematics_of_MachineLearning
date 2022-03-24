function [acc] = plot_preds(tit,X, t, preds)
%PLOT_PREDS Summary of this function goes here
%   Detailed explanation goes here

idx = (t > 0) & (t == preds); % true L1
plot( X(1,idx), X(2,idx), 'bx' ); hold on;
idx = (t < 0) & (t == preds); % true L2
plot( X(1,idx), X(2,idx), 'bo' );
idx = (t > 0) & (t.*preds < 0); % false L1
plot( X(1,idx), X(2,idx), 'rx' );
idx = (t < 0) & (t.*preds < 0); % false L2
plot( X(1,idx), X(2,idx), 'ro' );
title(tit);

acc = 100*sum(preds==t) / numel(t);

disp(['A ' tit ' gets ' num2str(acc) ' % correct']);
end

