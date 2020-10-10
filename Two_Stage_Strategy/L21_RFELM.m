function [Wi, H, H_test] = L21_RFELM( Traingdata,Trainglabel, Testingdata,Testinglabel ,Elm_Type, NumberofHiddenNeurons,C,Delta)

%% Parameter directions
% Usage: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = L21_RFELM( TrainData,TraingLabel, TestData,TestLabel,Elm_Type, NumberofHiddenNeurons,C,Delta)
%
% Input:
% TrainData     -  Training data matrix
%                         TrainData is a N*d matrix where N equals to the
%                         number of samples and d equals to the
%                         dimensionality of samples.
% TrainLabel    - Labels of training samples
% TrainData     -  Testing data matrix
% TrainLabel    - Labels of testing samples
% Elm_Type     - 0 for regression; 1 for (both binary and multi-classes) classification
% NumberofHiddenNeurons - Number of hidden neurons assigned to the L21_RFELM
% C                - Punishment coefficient
% Delta          - varience of hidden nodes parameters
% Output:
% Wi               - Selected hidden layer parameters
% H                 - Hidden layer matrix for training dataset
% H_test          - Hidden layer matrix for testing dataset
%%%%%%%%%%% Macro definition
REGRESSION=0;

%%%%%%%%%%% Load training dataset
%devide original data into training and testing
P = Traingdata';
T = Trainglabel';
clear Trainingdata Traininglabel

%%%%%%%%%%% Load testing dataset
TV.P = Testingdata';
TV.T = Testinglabel';
clear Testingdata Testinglabel;

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    
    %%%%%%%%%% Processing the targets of training
    % determine the category number
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break;
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;
    
    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break;
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;
    
end                                                 %   end if of Elm_Type

%%%%%%%%%%% Calculate weights & biases
% H hidden layer matrix
% W hidden layer parameters
[H,W] = rp_factorize(Traingdata,NumberofHiddenNeurons,'gaussian', Delta);
H = (1/sqrt(NumberofHiddenNeurons))*H;
%% L_21 Explicit Mapping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Objective_Value = zeros(1,100);
D_21_star = ones(size(H));
iter = 1;
while iter<35 && (iter<3 || abs(Objective_Value(iter-1)-Objective_Value(iter-2)) > Objective_Value(iter-1)*10^(-3))
    temp_D_H = D_21_star.*H;
    if NumberofTrainingData >= NumberofHiddenNeurons
        OutputWeight=(speye(size(H,1))/C+temp_D_H *H' ) \ temp_D_H* T';
    else
        OutputWeight = temp_D_H*((H'*temp_D_H + speye(size(H,2))/C)\T');
    end
    clear temp_D_H
    
    Wi = sqrt(sum(OutputWeight.*OutputWeight,2));
    d_21 = 2.*Wi;
    D_21_star = repmat(d_21,1,size(H,2));
    Y = (H' * OutputWeight)';
    Error = Y-T;
    Square_Error = norm(Error,'fro')^2;
    Objective_Value(iter) = (norm_2_1(OutputWeight)) + C*Square_Error;
    iter = iter+1;
end
fprintf('Iteration for L21_Norm is :%d\n',iter);
%````````````````````````````````````````````````
%%%%%%%%%%% Generate Testing Samples
H_test =(1/sqrt(NumberofHiddenNeurons))* rp_apply(TV.P,W);
