clc;clear all;

% Load the MATLAB file
load('LAA_no_slip_patient_4.mat');

% Extract variables
variable1 = C_star;
variable2 = P_star;
variable3 = U_star;
variable4 = V_star;
variable5 = x_star;
variable6 = y_star;
variable7 = t_star;

% Load the MATLAB file containing patch_ID
%load('LAA_no_slip_patient_2.mat','patch_ID'); 
%patch_ID = patch_ID; 

% Split each variable into smaller parts
split_size = 150;
num_splits = size(variable1, 2) / split_size;
for i = 1:num_splits
    start_index = (i - 1) * split_size + 1;
    end_index = i * split_size;
    % Modify the variable names accordingly
    C_star = variable1(:, start_index:end_index);
    P_star = variable2(:, start_index:end_index);
    U_star = variable3(:, start_index:end_index);
    V_star = variable4(:, start_index:end_index);
    x_star = variable5(:, start_index:end_index);
    y_star = variable6(:, start_index:end_index);
    t_star = variable7(start_index:end_index,:);

    % Save each split variable into a separate MATLAB file
    save(sprintf('split_LAA_no_slip_patient_4_150timesteps_%d.mat', i), 'C_star', 'P_star', 'U_star', 'V_star', 'x_star', 'y_star', 't_star','patch_ID');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clc;
% clear all;
% 
% % Load the MATLAB file
% load('split_LAA_no_slip_patient_4_150timesteps.mat');
% 
% % Extract variables
% variable1 = C_star;
% variable2 = P_star;
% variable3 = U_star;
% variable4 = V_star;
% variable5 = x_star;
% variable6 = y_star;
% variable7 = t_star;
% 
% % Load the MATLAB file containing patch_ID
% load('LAA_no_slip_patient_4.mat','patch_ID'); 
% patch_ID = patch_ID; 
% 
% % Define the percentage of data for training
% train_percentage = 0.8;
% 
% % Get the total number of data points
% num_data_points = size(variable1, 2);
% 
% % Generate random indices for selecting data
% random_indices = randperm(num_data_points);
% 
% % Calculate the number of data points for training
% num_train_data = round(train_percentage * num_data_points);
% 
% % Select training data
% train_indices = random_indices(1:num_train_data);
% test_indices = random_indices(num_train_data+1:end);
% 
% % Split data based on selected indices
% C_star_train = variable1(:, train_indices);
% P_star_train = variable2(:, train_indices);
% U_star_train = variable3(:, train_indices);
% V_star_train = variable4(:, train_indices);
% x_star_train = variable5(:, train_indices);
% y_star_train = variable6(:, train_indices);
% t_star_train = variable7(train_indices);
% 
% C_star_test = variable1(:, test_indices);
% P_star_test = variable2(:, test_indices);
% U_star_test = variable3(:, test_indices);
% V_star_test = variable4(:, test_indices);
% x_star_test = variable5(:, test_indices);
% y_star_test = variable6(:, test_indices);
% t_star_test = variable7(test_indices);
% 
% % Save training and test data into separate MATLAB files
% save('train_data_LAA_no_slip_patient_4.mat', 'C_star_train', 'P_star_train', 'U_star_train', 'V_star_train', 'x_star_train', 'y_star_train', 't_star_train','patch_ID');
% save('test_data_LAA_no_slip_patient_4.mat', 'C_star_test', 'P_star_test', 'U_star_test', 'V_star_test', 'x_star_test', 'y_star_test', 't_star_test','patch_ID');

