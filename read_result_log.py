# coding:utf-8

import os
import sys
import re
import csv
import torch
import numpy as np

def read_result():
    '''
    'args': args.__dict__,
    'v_iter': [],
    'abs_error': [],
    'rel_error': [],
    'true_y': [solution_numerical.squeeze().t()],
    'predict_y': [],
    'abs_error2': [],
    'rel_error2': [],
    'predict_y2': [],
    'abs_error3': [],
    'rel_error3': [],
    'predict_y3': [],
    'model_state_dict': [],
    'total_time': [],
    # add
    'A':[]
    'id_train':[]
    'id_test':[]
    'id_test2':[]
    'id_test3':[]
    'seed':[]
    'num_paras':[]    # 参数量
    'pred_result':
    't':
    '''
    path = r'results/HeatDynamics/grid/result_HeatDynamics_grid_0128-011309_0.pth'
    path = r'results/HeatDynamics/grid/result_HeatDynamics_grid_0128-011809_1.pth'
    results = torch.load(path)
    print(results['args']['activation_function'])
    print(results['args']['seed'])
    # print(results['true_y'][0].shape)
    # print(results['true_y'][0])
    # print(results['id_test'][0])
    # print(results['true_y'][0][:, 0].shape)
    # print(results['predict_y'][0].shape)
    # print(results['predict_y2'][0].shape)
    # print(results['rel_error'])
    # print(results['seed'][0])
    # print(results['t'])
    # print(results['model_state_dict'])
    # print(results['A'])
    # print(results['A'][0].shape)
    # for key in results.keys():
    #     print(key)


def read_file_name(file_dir):
    root = ""
    files = ""
    for root, dirs, files in os.walk(file_dir):
        print(root)
        print(dirs)
        print(files)
        pass
    return root, files

def read_log_RESULT(root, file_name_list, count, result_file_name):
    file_result_list = []
    file_result_list_count = []
    for i in range(len(file_name_list)):
        log_path = os.path.join(root, file_name_list[i])
        with open(log_path, "r", encoding='utf-8') as f:
            file = f.read()  # RESULT
            batch = re.split('RESULT ', file)
            batch = re.split('Time', batch[1])
            data = re.findall(r'2000\| Train Loss (.*?)\((.*?) Relative\) \| Test Loss (.*?)\((.*?) Relative\) \| Test Loss2 (.*?)\((.*?) Relative\) \| Test Loss3 (.*?)\((.*?) Relative.*?', batch[0], re.S)
            temp_list = [file_name_list[i]]
            temp_list.extend(list(data[0]))
            file_result_list_count.append(temp_list)

        if (i+1) % count == 0:
            result_array = np.zeros((count, 8))
            for ii in range(count):
                for jj in range(8):
                    result_array[ii][jj] = file_result_list_count[ii][jj+1]
            log_mean_name = [file_name_list[i-1][0:-6] + "_mean"]
            log_var_name = [file_name_list[i-1][0:-6] + "_std"]
            log_mean_var_name = [file_name_list[i - 1][0:-6] + "_mean_std"]
            log_mean = log_mean_name + np.mean(result_array, axis=0).tolist()
            # log_var = log_var_name + np.var(result_array, axis=0).tolist()
            log_std = log_var_name + np.std(result_array, axis=0).tolist()
            log_mean_three = log_mean_name + np.around(np.mean(result_array, axis=0), 3).tolist()
            log_var_three = log_var_name + np.around(np.std(result_array, axis=0), 3).tolist()
            log_mean_var_three = log_mean_var_name + [(str(log_mean_three[i]) + '+' + str(log_var_three[i])) for i in range(1, 9)]
            file_result_list_count.append(log_mean)
            file_result_list_count.append(log_std)
            file_result_list_count.append(log_mean_three)
            file_result_list_count.append(log_var_three)
            file_result_list_count.append(log_mean_var_three)
            # file_result_list_count.append(log_mean_var_three_percent)
            file_result_list += file_result_list_count
            file_result_list_count = []
            file_result_list.append([])
    # print(file_result_list)
    with open(result_file_name, 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['log', 'TrainLoss', 'TrainLossRelative', 'TestLoss', 'TestLossRelative', 'TestLoss2', 'TestLoss2Relative', 'TestLoss3', 'TestLoss3Relative'])
        for i in range(len(file_result_list)):
            csv_writer.writerow(file_result_list[i])
    pass

if __name__ == '__main__':
    # file_dir = r'.\log\HeatDynamics'
    # root, file_name_list = read_file_name(file_dir)
    # read_log_RESULT(root, file_name_list, 10, '.\log\DNND_log_heat_model_result.csv')
    #
    # file_dir = r'.\log\BiochemicalDynamics'
    # root, file_name_list = read_file_name(file_dir)
    # read_log_RESULT(root, file_name_list, 10, '.\log\DNND_log_biochemical_model_result.csv')
    #
    # file_dir = r'.\log\BirthDeathDynamics'
    # root, file_name_list = read_file_name(file_dir)
    # read_log_RESULT(root, file_name_list, 10, '.\log\DNND_log_birthdeath_model_result.csv')

    read_result()

