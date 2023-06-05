% load(['E:\code_and_data_package\PaperExperimentation20221216D\HeatDynamicsD' ...
%     '\create_mat_data\heat_data_of_model.mat'])
% model_data = model_data.'
% true_data_of_model = true_data.'
% true_lyapExp_of_model = lyapunovExponent(true_data_of_model, 100)
% model_lyapExp = lyapunovExponent(model_data, 100)
% 
% 
% load(['E:\code_and_data_package\PaperExperimentation20221216D\ndcn-masterD' ...
%     '\create_mat_data\heat_data_of_ndcn.mat'])
% true_data_of_ndcn = true_data_ndcn.'
% true_lyapExp_of_ndcn = lyapunovExponent(true_data_of_ndcn, 100)
% ndcn_lyapExp = lyapunovExponent(ndcn_data, 100)

% load('E:\code_and_data_package\PaperExperimentation20221216D\HeatDynamicsD\create_mat_data\result_HeatDynamics_grid_1217-010531_D0_mat_60_1200.mat');
% D0_true_lyapExp_of_model = lyapunovExponent(D0_true_data, 100);
% D0_pred_lyapExp_of_model = lyapunovExponent(D0_model_data, 100);

% dir_path = 'E:\code_and_data_package\PaperExperimentation20221216D\HeatDynamicsD\create_mat_data\';
dir_path = 'E:\code_and_data_package\PaperExperimentation20221216D\BirthDeathDynamicsD\create_mat_data\';
% dir_path = 'E:\code_and_data_package\PaperExperimentation20221216D\BiochemicalDynamicsD\create_mat_data\';
file_list = ls(dir_path);
model_lyapExp_cell = cell(1, 10);
true_lyapExp_of_model = cell(1, 10);
for i=4:13
    index = split(file_list(i, :), '_');
    index(5);
    file_path = strcat(dir_path, strtrim(file_list(i, :)));
    load(file_path);
    val_model_data = eval(strcat(index(5), "_model_data"));
    val_true_data = eval(strcat(index(5), "_true_data"));
    model_lyapExp_cell{i-3} = lyapunovExponent(val_model_data, 1000);
    true_lyapExp_of_model{i-3} = lyapunovExponent(val_true_data, 1000);
end


% dir_path = 'E:\code_and_data_package\PaperExperimentation20221216D\ndcn-masterD\create_mat_data\heat\';
% dir_path = 'E:\code_and_data_package\PaperExperimentation20221216D\ndcn-masterD\create_mat_data\birthdeath\';
% dir_path = 'E:\code_and_data_package\PaperExperimentation20221216D\ndcn-masterD\create_mat_data\biochemical\';
dir_path = 'E:\code_and_data_package\PaperExperimentation20221216D\ndcn-masterD-BirthDeath\create_mat_data\birthdeath\';
file_list = ls(dir_path);
ndcn_lyapExp_cell = cell(1, 10);
true_lyapExp_of_ndcn = cell(1, 10);
for i=4:13
    index = split(file_list(i, :), '_');
    index(5);
    file_path = strcat(dir_path, strtrim(file_list(i, :)));
    load(file_path);
    val_model_data = eval(strcat(index(5), "_ndcn_data"));
    val_true_data = eval(strcat(index(5), "_true_data"));
    ndcn_lyapExp_cell{i-3} = lyapunovExponent(val_model_data, 1000);
    true_lyapExp_of_ndcn{i-3} = lyapunovExponent(val_true_data, 1000);
end


