%matrix setup
a = magic(5);
b = a';
x = [a(:,1:3); b(:,1:3)];%[m=10 n=3]
%fprintf('x:\n'); disp(x);

%vectorised approach
sigma = x' * x;
fprintf('sigma:\n'); disp(sigma);

%iterative approach
xt = x';
m = size(x, 1);
sigmb = zeros(size(sigma));

for i = 1:m
    sigmb = sigmb + xt(:,i) * x(i,:);
end

fprintf('sigmb:\n'); disp(sigmb);