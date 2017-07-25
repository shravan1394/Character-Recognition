function p = predict(Theta0,Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
B=Theta0(1,:);
Theta0=Theta0(2:end,:);
Theta0=Theta0(:);
q=(0:20:300)'*ones(1,16);
q=q';
q=q(:)*ones(1,400);
t=(100:20:400)'*ones(1,16);
t=t';
t=t(:)*ones(1,400);
f=(0:15)'*ones(1,16);
f=f(:)*ones(1,400);
Y=0:399;
a=((rem(Y,20)'*ones(1,256))'<(f+5) & (rem(Y,20)'*ones(1,256))'>=f & (Y'*ones(1,256))'>=q & (Y'*ones(1,256))'<t);
a=repmat(a,20,1);
a1=logical(a);
a=logical(a');
a=a(:);
r=reshape(Theta0,25,[]);
r=repmat(r,256,1);
r=r(:);
W=zeros(400*256*20,1);
W(a)=r;
W=reshape(W,400,[])';
X1=X;
B1=B'*ones(1,256);
B1=B1';
B1=B1(:);
X=sigmoid([ones(size(X,1),1) X]*[B1';W']);
%X=reshape(X',2,[]);
%X=max(X);
%X=reshape(X,8,[]);
%X=reshape(X',2,[]);
%X=max(X);
%X=reshape(X,[],8)';
%X=reshape(X,64,[])';
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
%for i=num_labels:-1:1
%	A(i)=2^i
	
[dummy, p] = max(h2, [], 2);
%[char(p+64) num2str(dummy) num2str(mean(h2,2)) num2str(mean(h2,2)+std(h2,[],2)*1) num2str(mean(h2,2)+std(h2,[],2)*2) num2str(mean(h2,2)+std(h2,[],2)*3) ]
%h2

save p.mat p
% =========================================================================


end
