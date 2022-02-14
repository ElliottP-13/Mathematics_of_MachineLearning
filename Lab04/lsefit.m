%% Least-Squared Error FIT
%  Find the linear combination of basis functions which best model the data.
%
%  Inputs:
%  
%  x - Vector with observation locations in 1D. (indep. variable)
%  t - Vector with observations in 1D. (dep. variable)
%  params - Parameters for the basis functions to be used in func, e.g. as
%    produced by gauss_basis.
%  func - Function handle which evaluates a basis function with parameters
%    given by the columns of params and at the specified locations. e.g. 
%    @gauss_basis, or @hat_basis.
%    For example, the first basis function at x = 2 is func(2, params(:,1)).
%  mu - Scalar representing the standard deviation of the prior Gaussian on
%    the model parameters.
%
%  Outputs:
%  w - Coefficients used to generate a linear combination of the basis 
%    functions which is the maximum likelihood learned model.

function [w] = lsefit(x, t, params, func, mu)

end
