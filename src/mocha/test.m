clc, clear
%% load dataset
Xtrain = load('./data/HAR/raw/train/X_train.txt');
Ytrain = load('./data/HAR/raw/train/y_train.txt');
Strain = load('./data/HAR/raw/train/subject_train.txt');
Xtest = load('./data/HAR/raw/test/X_test.txt');
Ytest = load('./data/HAR/raw/test/y_test.txt');
Stest = load('./data/HAR/raw/test/subject_test.txt');

X = [Xtrain; Xtest];
Y = [Ytrain; Ytest];
S = [Strain; Stest];
All = [S, Y, X];
All = sortrows(All, 1);
Y = All(:,2);
X = All(:,3:end);
ans = tabulate(S); rowDist = ans(:,2);
X = mat2cell(X, rowDist)';
Y = mat2cell(Y, rowDist)';

%% set parameters
addpath('./src/mocha/opt/'); addpath('./src/mocha/util/'); % add helper functions
ntrials = 1; % number of trials to run
training_percent = 0.75; % percentage of data for training
opts.obj='C'; % classification
opts.avg = 1; % compute average error across tasks

%% set hyperparameter search space
lambda_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 10]; % regularizer
rng(1);
[Xtrain, Ytrain, Xtest, Ytest] = split_data(X, Y, training_percent);

%% MTL model (mocha)
opts.mocha_outer_iters = 10;
opts.mocha_inner_iters = 100;
opts.mocha_sdca_frac = 0.5;
opts.w_update = 0; % do a full run, not just one w-update
opts.sys_het = 0; % not messing with systems heterogeneity
mocha_lambda = cross_val_1(Xtrain, Ytrain, 'run_mocha', opts, lambda_range, 5); % determine via 5-fold cross val
[rmse_mocha_reg, primal_mocha_reg, dual_mocha_reg] = run_mocha(Xtrain, Ytrain, Xtest, Ytest, mocha_lambda, opts);
rmse_mocha_reg(end)