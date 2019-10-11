%% decision tree for diagnosis of breast cancer

%% Empty environment variables
clear all
clc
warning off

%% input data
load data.mat
% randomly generate training/testing datasets
a = randperm(569);
Train = data(a(1:500),:);
Test = data(a(501:end),:);
% training datasets
P_train = Train(:,3:end);
T_train = Train(:,2);
% testing datasets
P_test = Test(:,3:end);
T_test = Test(:,2);

%% Create a decision tree classifier
ctree = ClassificationTree.fit(P_train,T_train);
% check the decision tree view
view(ctree);
view(ctree,'mode','graph');

%% simulation testing
T_sim = predict(ctree,P_test);

%% results analysis
count_B = length(find(T_train == 1));
count_M = length(find(T_train == 2));
rate_B = count_B / 500;
rate_M = count_M / 500;
total_B = length(find(data(:,2) == 1));
total_M = length(find(data(:,2) == 2));
number_B = length(find(T_test == 1));
number_M = length(find(T_test == 2));
number_B_sim = length(find(T_sim == 1 & T_test == 1));
number_M_sim = length(find(T_sim == 2 & T_test == 2));
disp(['total number of cases£º' num2str(569)...
      '  benign cases£º' num2str(total_B)...
      '  malignant cases£º' num2str(total_M)]);
disp(['number of cases in training dataset£º' num2str(500)...
      '  benign cases£º' num2str(count_B)...
      '  malignant cases£º' num2str(count_M)]);
disp(['number of cases in testing dataset£º' num2str(69)...
      '  benign cases£º' num2str(number_B)...
      '  malignant cases£º' num2str(number_M)]);
disp(['confirmed benign cases£º' num2str(number_B_sim)...
      '  misdiagnose£º' num2str(number_B - number_B_sim)...
      '  correct diagnosis rate p1=' num2str(number_B_sim/number_B*100) '%']);
disp(['confirmed malignant cases£º' num2str(number_M_sim)...
      '  misdiagnose£º' num2str(number_M - number_M_sim)...
      '  correct diagnosis rate p2=' num2str(number_M_sim/number_M*100) '%']);
  
%% effects of minimum sample amount contained in the leaf node on the performance of decision tree
leafs = logspace(1,2,10);

N = numel(leafs);

err = zeros(N,1);
for n = 1:N
    t = ClassificationTree.fit(P_train,T_train,'crossval','on','minleaf',leafs(n));
    err(n) = kfoldLoss(t);
end
plot(leafs,err);
xlabel('minimum sample amount contained in the leaf node');
ylabel('cross-validated relative error');
title('effects of minimum sample amount contained in the leaf node on the performance of decision tree')

%% set minleaf to 28£¬generate optimized decision tree
OptimalTree = ClassificationTree.fit(P_train,T_train,'minleaf',28);
view(OptimalTree,'mode','graph')

resubOpt = resubLoss(OptimalTree)
lossOpt = kfoldLoss(crossval(OptimalTree))
resubDefault = resubLoss(ctree)
lossDefault = kfoldLoss(crossval(ctree))

%% cut branches
[~,~,~,bestlevel] = cvLoss(ctree,'subtrees','all','treesize','min')
cptree = prune(ctree,'Level',bestlevel);
view(cptree,'mode','graph')

resubPrune = resubLoss(cptree)
lossPrune = kfoldLoss(crossval(cptree))

