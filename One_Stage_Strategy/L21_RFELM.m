function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = L21_RFELM( TrainData,TrainLabel, TestData,TestLabel,Elm_Type, NumberofHiddenNeurons,C,Delta)
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
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy:
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy:
%                           RMSE for regression or correct classification rate for classification
%
%%%%%%%%%%% 
REGRESSION=0;
CLASSIFIER=1;

%%%%%%%%%%% Load training dataset
%devide original data into training and testing
P = TrainData';
T = TrainLabel';
clear Trainingdata Traininglabel

%%%%%%%%%%% Load testing dataset
TV.P = TestData';
TV.T = TestLabel';
clear Testingdata Testinglabel;

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    % determine the category number
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               
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
end                                                 
%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;

%%%%%%%%%%% Randomly generate the hidden layer parameters
% H hidden layer matrix
% W hidden layer parameters
[H,W] = rp_factorize(TrainData,NumberofHiddenNeurons, 'gaussian', Delta);
H = (1/sqrt(NumberofHiddenNeurons))*H;
%% L_21 Explicit Mapping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Objective_Value = zeros(1,35);
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Find those dimensions which contribute more than 96% and less than 96.5% of the total information
Wei_OP = sum(abs(OutputWeight),2);
Total_Weight = sum(Wei_OP);
[Val, Secquence_1] = sort(Wei_OP,'descend');
percentage = 0;
Start_Point = 1;
End_Point = length(Val);
while percentage<0.96 || percentage>0.965
    Feature_Num = ceil((Start_Point+End_Point)/2);
    percentage = sum(Val(1 : ceil((Start_Point+End_Point)/2)))/Total_Weight;
    if  percentage > 0.965
        End_Point = ceil((Start_Point+End_Point)/2);
    else
        Start_Point = ceil((Start_Point+End_Point)/2);
    end
    if abs(End_Point - Start_Point)<=2
        break
    end
end

Selected_Features = Secquence_1(1:Feature_Num);
%````````````````````````````````````````````````
OutputWeight = OutputWeight(Selected_Features,:);
H = H(Selected_Features,:);
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;        %   Calculate CPU time (seconds) spent for training ELM
%%%%%%%%%%% Calculate the training accuracy
Y=real(H' * OutputWeight)';                             %   Y: the actual output of the training data
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y));               %   Calculate training accuracy (RMSE) for regression case
end
clear H;

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
H_test =(1/sqrt(NumberofHiddenNeurons))* rp_apply(TV.P,W);
H_test = H_test(Selected_Features,:);
TY=real(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
TestingTime = TestingTime*length(Selected_Features)/(4096*2);


if Elm_Type == REGRESSION
    TestingAccuracy=sqrt(mse(TV.T - TY));            %   Calculate testing accuracy (RMSE) for regression case
end

if Elm_Type == CLASSIFIER
    %%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;
    
    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected;
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2);
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);
end