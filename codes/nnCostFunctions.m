function [J,grad] = nnCostFunctions(nn_params,gn, ...
                                   Theta0,Theta1,Theta2,M,N,input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)



%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshaping nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for the 2 layer neural network
Theta0 = reshape(Theta0,M+1,N);
%Theta0=reshape(nn_params,M,N);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%Theta1 = reshape(Theta1, ...
                % hidden_layer_size, (input_layer_size + 1));

%Theta2 = reshape(Theta2, ...
                 %num_labels, (hidden_layer_size + 1));
X1=X;
y1=y;
m = size(X1, 1);
        
% Initializing 
J = 0;
Theta0_grad = zeros(size(Theta0));
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Feedforward the neural network and return the cost in the variable J. 
B=Theta0(1,:);
Theta0=Theta0(2:end,:);
Theta0=Theta0(:);
d1=0;d2=0;d3=0;
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
a1=logical(a);
a=repmat(a,20,1);
a=logical(a');
a=a(:);
a1=a1';
a1=a1(:);
r=reshape(Theta0,25,[]);
r=repmat(r,256,1);
r=r(:);
W=zeros(400*256*20,1);
Z=zeros(size(a1));
W(a)=r;
e=((2:26)'*ones(1,256));
Z(a1)=e(:);
Z=reshape(Z,400,[])';
W=reshape(W,400,[])';
X2=X1;
B1=B'*ones(1,256);
B1=B1';
B1=B1(:);
X1=sigmoid([ones(size(X1,1),1) X1]*[B1';W']);
%X=reshape(X',2,[]);
%X=max(X);
%X=reshape(X,8,[]);
%X=reshape(X',2,[]);
%X=max(X);
%X=reshape(X,[],8)';
%X=reshape(X,64,[])';
y1=double(y1);
y1=dec2bin(2.^(num_labels-y1),num_labels);
y1=y1-48;
h0=sigmoid([ones(size(X1,1),1) X1]*Theta1'); 
h=sigmoid([ones(size(h0,1),1) h0]*Theta2');
J=-(1/m)*sum(sum(y1.*log(h)+(ones(size(y1))-y1).*log(ones(size(h))-h)))+(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
%J=sum(sum((h-y).*(h-y)))/m ;

% Implement the backpropagation algorithm to compute the gradients
% Theta1_grad and Theta2_grad. 

d3=h-y1;
d2=d3*Theta2(:,2:end).*sigmoidGradient([ones(size(X1,1),1) X1]*Theta1');
Theta2_grad=(((d3'*[ones(size(h0,1),1) h0]))/m)+(lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)];
Theta1_grad=(((d2'*[ones(size(X1,1),1) X1]))/m)+(lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
%d1=d2*Theta1(:,2:end);
%d1=d1';
%d1=d1(:)*ones(1,4);
%d1=reshape(d1',256,[])'.*sigmoidGradient(X1*W');
%d1=d1.*sigmoidGradient([ones(size(X2,1),1) X2]*[B1';W']);
%d1=reshape(d1',256,[]);
%for b=1:256
%	Aa=(a1(b,:)'*ones(1,size(X1,1)))';
%	g=X1.*Aa;
%	Aa=Aa';
 %   Aa=logical(Aa(:));
%	g=g';
%	g=g(:);
%	g=reshape(g(Aa),25,[]);
%	d1(1+25*(b-1):25+25*(b-1),:)=d1(1+25*(b-1):25+25*(b-1),:).*g;
%	su=su+sum(d1(1+25*(b-1):25+25*(b-1),:),2);
%end
%Aa=1:(size(X1,1)*20);
%Aa=reshape(Aa,20,[]);
%Aa=Aa';
%Aa=Aa(:)';
%for b=1:26
%    if b==1
%        su=sum(reshape(d1(:,Aa),256*size(X1,1),[]));
%        Th=[Th;su];
%    else
%        S=logical(((max(Z==b))'*ones(1,size(X1,1)))');
%        g=X1';
%        g=g(S');
%        Su=g'*reshape(d1(:,Aa),256*size(X1,1),[]);
%        Th=[Th;Su];
%    end
%end    
%Th=Th/m;
%Theta0_grad=Th;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
