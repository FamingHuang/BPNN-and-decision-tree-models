%% prediction based on BPNN model
%

%% Empty environment variables
clc
clear

%% extraction of training and testing data and normalization 
%input and output data
load data input output

%random sort from 1 to 2000
k=rand(1,2000);
[m,n]=sort(k);

%find out the training and testing data
input_train=input(n(1:1900),:)';
output_train=output(n(1:1900));
input_test=input(n(1901:2000),:)';
output_test=output(n(1901:2000));

%data normalization
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%% training of BP neural network
% %Initialization of network structure
net=newff(inputn,outputn,5);

net.trainParam.epochs=100;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00004;

%network training
net=train(net,inputn,outputn);

%% prediction of BPNN
%normalization of prediction data
inputn_test=mapminmax('apply',input_test,inputps);
 
%output of BPNN model
an=sim(net,inputn_test);
 
%normalization of output
BPoutput=mapminmax('reverse',an,outputps);

%% results analysis

figure(1)
plot(BPoutput,':og')
hold on
plot(output_test,'-*');
legend('predictive results','expected results')
title('predictive output of BPNN','fontsize',12)
ylabel('output of function','fontsize',12)
xlabel('sample','fontsize',12)
%predictive errors
error=BPoutput-output_test;


figure(2)
plot(error,'-*')
title('predictive errors','fontsize',12)
ylabel('errors','fontsize',12)
xlabel('samples','fontsize',12)

figure(3)
plot((output_test-BPoutput)./BPoutput,'-*');
title('Percentage of predictive errors of BPNN')

errorsum=sum(abs(error));

web browser www.matlabsky.com
%%
