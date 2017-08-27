load('data.mat')
Y=0:1016*26-1;
load('data.mat');
X=reshape(X,62992,400);
X=X(10161:36576,:);
X=X(rem(Y,1016)<508,:);
C=X;
% finding the covariance of the entire dataset
CX=((C-mean(C,2)*ones(1,size(C,2)))*(C-mean(C,2)*ones(1,size(C,2)))')/size(C,2);
%size(CX)
% Finding the eigen vectors of the covariance matrix
[E,~]=eig(CX);
e=eig(CX);
plot(e)
% Transforming/Projecting the original data into lower dimensional space
U=E(:,2125:2304)'*C;
e(2124)
E=E(:,2125:2304);
%save E.mat E
