function [value]=norm_2_1(A)
[row]=size(A,1);
value=0;
for i=1:row
    value=value+norm(A(i,:));
end
