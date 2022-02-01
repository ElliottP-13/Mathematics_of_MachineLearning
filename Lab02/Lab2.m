clc; clear;
%% Part 1
% In this part, we were first asked to fix the gaussfit function.
% We are then asked to load the employees.mat file and use gaussfit to
% better understand the salaries of different departments.

% First we show that gaussfit works. I assume gaussfit returns [mu, sigma^2]
% where mu = mean, sigma^2 = variance. There was conflicting information
% in MATLAB vs PDF assignment as to which one to return, and returning the 
% variance maintains higher numerical precision.

mu_s = [1 2 3 4 5];
sigma_s = 3 * eye(size(mu_s, 2));
n = 10000;
X = mvnrnd(mu_s,sigma_s,n)';  % n random samples from multivariate gaussian
[mu, sigma] = gaussfit(X)

% We see that the output matches what we expect. We expected to find that
% mu = [1 2 3 4 5] and sigma = 3. We get pretty close :)
% Next we load in the employee data
load("employees.mat");

max_dept = max(dept(:));  % get the maximum department number for iterating
mu = zeros(max_dept,1); % initialize matrices
sigma = zeros(max_dept,1);
for i = 1:max_dept
    [mu(i),sigma(i)] = gaussfit( sal(dept == i) );
end

% Now we find the departments that have the highest and lowest mean salary
[~, min_mu] = min(mu);
[~, max_mu] = max(mu);
deptnames = fieldnames(depts);  % store strings of names of the departments
deptnumbers = struct2cell(depts);
deptnumbers = [deptnumbers{:}];
disp("Dept with lowest mean = " + min_mu + " (" + deptnames(deptnumbers==min_mu) + "), mu = $" + round(mu(min_mu), 2));
disp("Dept with highest mean = " + max_mu + " (" + deptnames(deptnumbers==max_mu) + "), mu = $" + round(mu(max_mu), 2));
