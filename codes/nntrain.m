%% Initialization
clear ; close all; clc
A=[];
B=[];
H=[];
D=[];
gn=0;
%% Setup the parameters you will use for this exercise
input_layer_size  = 256*20;  % Input Images of faces after PCA
hidden_layer_size = 100;   % 60 hidden units
num_labels = 26;          % 40 labels, from 1 to 40   
M=25;
N=20;
                          

%% =========== Part 1: Loading Data =============

% Load Training Data
fprintf('Loading Data ...\n')
y1=1:26;
y1=(y1'*ones(1,1016))';
y1=y1(:);
Y=0:1016*26-1;
y1=y1(rem(Y,1016)>00 & rem(Y,1016)<=508);
load('data.mat');
%load Ed
y=y1;
X=reshape(X,62992,400);
X=X(10161:36576,:);
%X=X*E;
size(X)
X=X(rem(Y,1016)>00 & rem(Y,1016)<=508,:);
m = size(X, 1);
%36576
fprintf('Program paused. Press enter to continue.\n');
pause;



%% ================ Part 2: Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')
%initial_Theta0 = randInitializeWeights(M, N);
%initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
%initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
%size(initial_Theta1)
load tr1.mat
load tr2.mat
load tr0.mat
initial_Theta1=Theta1;
initial_Theta2=Theta2;
initial_Theta0=Theta0;
%pred = predict(Theta0,Theta1, Theta2, X);

%fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred == y)) * 100);
% Unroll parameters
nn_params = [initial_Theta0(:);initial_Theta1(:) ; initial_Theta2(:)];



%% =================== Part 3: Training NN ===================

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 00);

L=[4.5,5.2];
%lambda =5; %2.0 2.4 8.05
for lambda=0.1
fprintf('Loading Data ...\n')
y1=1:26;
y1=(y1'*ones(1,1016))';
y1=y1(:);
Y=0:1016*26-1;
y1=y1(rem(Y,1016)>00 & rem(Y,1016)<=508);
load('data.mat');
y=y1;
X=reshape(X,62992,400);
X=X(10161:36576,:);
%X=X*E;
X=X(rem(Y,1016)>00 & rem(Y,1016)<=508,:);
% Create "short hand" for the cost function to be minimized
costFunction = @(p,gn) nnCostFunction(p,gn, ...
                                   M,N,input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
                               
lambda
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
%[nn_params, cost] = fmincg(costFunction, initial_nn_params,gn, options);
%[nn_params, cost] = descent(costFunction, initial_nn_params,gn, 00);

% Obtain Theta1 and Theta2 back from nn_params
Theta0 = reshape(nn_params(1:(M+1) * N),M+1,N);
Theta1 = reshape(nn_params(1+(M+1)*N:(M+1)*N+(hidden_layer_size * (input_layer_size + 1))), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + ((M+1)*N+hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

A=[A; Theta1];
B=[B;Theta2];
fprintf('Program paused. Press enter to continue.\n');
%pause;


%% ================= Part 4: Prediction =================
%  After training the neural network, we would like to use it to predict
%  the labels. This lets us compute the training set accuracy.

pred = predict(Theta0,Theta1, Theta2, X);
%stem(pred-y);
%title ('Plot of difference between label and predicted class of training dataset');
%figure;
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
H=[H;mean(double(pred == y)) * 100]
fprintf('Loading Data ...\n')
y1=1:26;
y1=(y1'*ones(1,1016))';
y1=y1(:);
Y=0:1016*26-1;
y1=y1(rem(Y,1016)>=508);
load('data.mat');
y=y1;
X=reshape(X,62992,400);
X=X(10161:36576,:);
%X=X*E;
X=X(rem(Y,1016)>=508,:);

%wuiogjlcvb
fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ================= Part 2: Prediction =================
%  After training the neural network, we would like to use it to predict
%  the labels. This lets us compute the testing set accuracy.

pred = predict(Theta0,Theta1, Theta2, X);

fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred == y)) * 100);
D=[D;mean(double(pred == y)) * 100]
%stem(pred-y);
%title ('Plot of difference between label and predicted class of testing dataset');
end
%save tr111.mat Theta1
%save tr222.mat Theta2
%save tr000.mat Theta0
