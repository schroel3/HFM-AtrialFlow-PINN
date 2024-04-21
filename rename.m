
% Load the MAT file
load('train_data_LAA_no_slip_patient_3.mat');

% Rename variables as needed
C_star = C_star_train;
P_star = P_star_train;
U_star= U_star_train;
V_star = V_star_train;
x_star = x_star_train;
y_star = y_star_train;
t_star = t_star_train;
patch_ID;


% C_star = C_star_test;
% P_star = P_star_test;
% U_star= U_star_test;
% V_star = V_star_test;
% x_star = x_star_test;
% y_star = y_star_test;
% t_star = t_star_test;

% Save the modified variables back into the MAT file
save('train_data_LAA_no_slip_patient_3.mat','C_star','P_star','U_star','V_star','x_star','y_star','t_star','patch_ID');