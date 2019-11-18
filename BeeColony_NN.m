clc
tic 
close all 
clear all 
  

inputs=xlsread('inputlag1lag10.xlsx')';
targets=xlsread('targetlag1lag10.xlsx')';

ni=length(targets);
test=round(ni*0.25);


inputs_train=inputs(:,1:ni-test-1);
targets_train=targets(1:ni-test-1);

inputs_test=inputs(:,ni-test:ni);
targets_test=targets(ni-test:ni);

[InputTrainN,meanITrain,stdITrain,TargetTrainN,meanTTrain,stdTTrain]=prestd(inputs_train,targets_train);
[InputTestN,meanITest,stdITest,TargetTestN,meanTTest,stdTTest]=prestd(inputs_test,targets_test);

m=length(InputTrainN(:,1)); 
o=length(TargetTrainN(:,1)); 
  
n=2; 
net=feedforwardnet(n);
net=configure(net,InputTrainN,TargetTrainN);
net.layers{1}.transferFcn='tansig';
kk=m*n+n+n+o; 

for j=1:kk 
    LB(1,j)=-1; 
    UB(1,j)= 1; 
end  

pop=10;
nOnLooker=pop;
for i=1:pop 
    for j=1:kk 
        xx(i,j)=LB(1,j)+rand*(UB(1,j)-LB(1,j)); 
    end 
end
C=zeros(pop,1);
maxrun=3; 
for run=1:maxrun 
    fun=@(x) myfunc(x,n,m,o,net,InputTrainN,TargetTrainN); 
    x0=xx; 
 
    % inisialisasi MABC-----------------------------------------------mulai  
    p=x0;                   % populasi awal  
    for i=1:pop 
        f0(i,1)=fun(p(i,:)); 
    end
    [fmin0,index0]=min(f0);
    bestp=p(index0,:);
    bestc=fmin0;
    % inisialisasi MABC---------------------------------------------selesai 
    % algoritma MABC--------------------------------------------------mulai 
    ite=1; maxite=1000; L=round(0.6*kk*pop); tolerance=1; 
    while ite<=maxite && tolerance>10^-8     
        % Recruited Bees 
        for i=1:pop 
            for j=1:kk
                % Choose k randomly, not equal to i
                K=[1:i-1 i+1:pop];
                k=K(randi([1 numel(K)]));
                phi=unifrnd(-1,+1);             
                varphi=unifrnd(0,1.5);
                % New Bee Position
                newbee(i,j)=p(i,j)+phi*(p(i,j)-p(k,j))+varphi*(bestp(1,j)-p(i,j));
            end
        end
        %evaluation
        for i=1:pop
            f(i,1)=fun(newbee(i,:));
        end
        %Comparision
        for i=1:pop
            if f(i,1)<=f0(i,1)
                p(i,:)=newbee(i,:);
                f0(i,1)=f(i,1);
            else
                C(i)=C(i)+1;
            end
        end        
        % Menghitung Nilai Fitness dan Probabilitas
        MeanCost=mean(f0);
        F=calculateFitness(f0); 
        rho=2.5;
        P=exp(-(1/rho)*F);
        % Onlooker Bees
        for on=1:nOnLooker
            for j=1:kk
                % Select Source Site
                i=RouletteWheelSelectionM(f0);
                % Choose k randomly, not equal to i
                K=[1:i-1 i+1:pop];
                k=K(randi([1 numel(K)]));
                phi=unifrnd(-1,+1);             
                varphi=unifrnd(0,1.5);
                % New Bee Position
                newbee(i,j)=p(i,j)+phi*(p(i,j)-p(k,j))+varphi*(bestp(1,j)-p(i,j));
            end
        end
        %evaluation
        for i=1:nOnLooker
            f(i,1)=fun(newbee(i,:));
        end
        %Comparision
        for i=1:nOnLooker
            if f(i,1)<=f0(i,1)
                p(i,:)=newbee(i,:);
                f0(i,1)=f(i,1);
            else
                C(i)=C(i)+1;
            end
        end
        % Scout Bees
        for i=1:pop
            if C(i)>=L
                for ik=1:pop
                    for jk=1:kk 
                        pr(ik,jk)=LB(1,jk)+rand*(UB(1,jk)-LB(1,jk)); 
                    end
                end
                p=pr;
                for d=1:pop
                    f0r(d,1)=fun(pr(d,:));
                end
                f0=f0r;
                C(i)=0;
            end
        end
        % Update Best Solution Ever Found
         for i=1:pop
             if f0(i,1)<=bestc;
                 bestp=p(i,:);
                 bestc=f0(i,1);
             end
         end
        %Store Best Cost Ever Found
        fbcost(ite,run)=bestc;
        fbcostite(run)=ite;
        % calculating tolerance
        if ite>100; 
            tolerance=abs(fbcost(ite-100,run)-bestc); 
        end 
        %displaying iterative results
        if ite==1
            disp(sprintf('Iteration         Objective fun'));
        end
        disp(sprintf('%8g             %8.4f',ite,bestc));
        ite=ite+1;
    end
    %algoritma MABC-------------------------------------------------selesai
      
    xbest(run,:)=bestp; 
    ybest(run,1)=bestc;  
    disp(sprintf('****************************************'));
    disp(sprintf('    RUN            ObFuVa'));  
    disp(sprintf('%6g       %8.4f',run,ybest(run,1)));
end
toc

%final neural network model
disp('Final NN model is net_f')
net_f=feedforwardnet(n);
net_f=configure(net_f,InputTrainN,TargetTrainN);
net_f.layers{1}.transferFcn='tansig';
[a b]=min(ybest);
xo=xbest(b,:);
k=0;
for i=1:n 
    for j=1:m 
        k=k+1; 
        xi(i,j)=xo(k); 
    end 
end 
for i=1:n 
    k=k+1; 
    xl(i)=xo(k); 
    xb1(i,1)=xo(k+n); 
end 
for i=1:o 
    k=k+1; 
    xb2(i,1)=xo(k); 
end 
net_f.iw{1,1}=xi; 
net_f.lw{2,1}=xl; 
net_f.b{1,1}=xb1; 
net_f.b{2,1}=xb2; 

output_train=poststd(net_f(InputTrainN),meanTTrain,stdTTrain);
output_test=poststd(net_f(InputTestN),meanTTest,stdTTest);

%Calculation of MSE Training
mse_train=sum((output_train-targets_train).^2)/length(output_train)

%Calculation of MSE Testing
mse_test=sum((output_test-targets_test).^2)/length(output_test)

%Calculation of MAPE Training
errors_train=gsubtract(targets_train,output_train);
mape_train=mean(abs(errors_train./targets_train))

%Calculation of MAPE Testing
errors_test=gsubtract(targets_test,output_test);
mape_test=mean(abs(errors_test./targets_test))
  
%Plot Training
figure(1)
t=[1:length(targets_train)];
plot(t,targets_train,'k',t,output_train,'b')
xlim([0 length(targets_train)])

%Plot Test
figure(2)
t=[1:length(targets_test)];
plot(t,targets_test,'k',t,output_test,'b')
xlim([0 length(targets_test)])
disp('Trained ANN net_f is ready for the use');