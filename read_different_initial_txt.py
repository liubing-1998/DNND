# coding:utf-8

import re
import csv
import sys

import torch
import numpy as np

def read_different_initial_txt(file_path, count):
    with open(file_path, "r") as f:
        file = f.read()
        batch = re.split('============================================', file)
        result_list = []
        result_list_count = []
        for i in range(1, len(batch)):
            # print(item)
            batch_line = batch[i].split('\n')  # 取值batch_line[1]和batch_line[8]
            log_name = batch_line[1].split(' ')[-1]
            # print(log_name)
            data = re.findall(r'Test Loss1 (.*?)\((.*?) Relative\) \| Test Loss2 (.*?)\((.*?) Relative\) \| Test Loss3 (.*?)\((.*?) Relative\).*?', batch_line[8].split('RESULT ')[-1], re.S)
            # print(data)
            temp_list = [log_name]
            temp_list.extend(list(data[0]))
            # print(temp_list)
            result_list_count.append(temp_list)
            # sys.exit()

            if i % count == 0:
                result_array = np.zeros((count, 6))  # count行，8列结果
                for ii in range(count):
                    for jj in range(6):
                        result_array[ii][jj] = result_list_count[ii][jj + 1]
                mean_name = [result_list_count[0][0].split("__")[0] + "__mean"]
                var_name = [result_list_count[0][0].split("__")[0] + "__std"]
                mean_var_name = [result_list_count[0][0].split("__")[0] + "__mean_std"]
                mean = mean_name + np.mean(result_array, axis=0).tolist()
                std = var_name + np.std(result_array, axis=0).tolist()
                mean_three = mean_name + np.around(np.mean(result_array, axis=0), 3).tolist()
                var_three = var_name + np.around(np.std(result_array, axis=0), 3).tolist()
                mean_var_three = mean_var_name + [(str(mean_three[i]) + '+' + str(var_three[i])) for i in range(1, 7)]
                result_list_count.append(mean)
                result_list_count.append(std)
                result_list_count.append(mean_three)
                result_list_count.append(var_three)
                result_list_count.append(mean_var_three)
                result_list += result_list_count
                result_list_count = []
                result_list.append([])

        with open('different_initial_interpolation_extrapolation_longtime/different_initial_all.csv', 'a', encoding='utf-8', newline='') as f:
            # 2. 基于文件对象构建 csv写入对象
            csv_writer = csv.writer(f)
            # 3. 构建列表头
            csv_writer.writerow(['name', 'TestLoss1', 'TestLoss1Relative', 'TestLoss2', 'TestLoss2Relative', 'TestLoss3', 'TestLoss3Relative'])
            # 4. 写入csv文件内容
            for i in range(len(result_list)):
                csv_writer.writerow(result_list[i])

def read_different_A_initial_txt(file_path, count):
    with open(file_path, "r") as f:
        file = f.read()
        batch = re.split('============================================', file)
        result_list = []
        result_list_count = []
        for i in range(1, len(batch)):
            # print(item)
            batch_line = batch[i].split('\n')  # 取值batch_line[1]和batch_line[10]
            # for i in range(len(batch_line)):
            #     print(i, batch_line[i])
            # sys.exit()
            log_name = batch_line[1].split(' ')[-1]
            # print(log_name)
            data = re.findall(r'Test Loss1 (.*?)\((.*?) Relative\) \| Test Loss2 (.*?)\((.*?) Relative\) \| Test Loss3 (.*?)\((.*?) Relative\).*?', batch_line[10].split('RESULT ')[-1], re.S)
            # print(data)
            temp_list = [log_name]
            temp_list.extend(list(data[0]))
            # print(temp_list)
            result_list_count.append(temp_list)
            # sys.exit()

            if i % count == 0:
                result_array = np.zeros((count, 6))  # count行，8列结果
                for ii in range(count):
                    for jj in range(6):
                        result_array[ii][jj] = result_list_count[ii][jj + 1]
                mean_name = [result_list_count[0][0].split("__")[0] + "__mean"]
                var_name = [result_list_count[0][0].split("__")[0] + "__std"]
                mean_var_name = [result_list_count[0][0].split("__")[0] + "__mean_std"]
                mean = mean_name + np.mean(result_array, axis=0).tolist()
                std = var_name + np.std(result_array, axis=0).tolist()
                mean_three = mean_name + np.around(np.mean(result_array, axis=0), 3).tolist()
                var_three = var_name + np.around(np.std(result_array, axis=0), 3).tolist()
                mean_var_three = mean_var_name + [(str(mean_three[i]) + '+' + str(var_three[i])) for i in range(1, 7)]
                result_list_count.append(mean)
                result_list_count.append(std)
                result_list_count.append(mean_three)
                result_list_count.append(var_three)
                result_list_count.append(mean_var_three)
                result_list += result_list_count
                result_list_count = []
                result_list.append([])

        with open('different_A_initial_interpolation_extrapolation_longtime/different_A_initial_all.csv', 'a', encoding='utf-8', newline='') as f:
            # 2. 基于文件对象构建 csv写入对象
            csv_writer = csv.writer(f)
            # 3. 构建列表头
            csv_writer.writerow(['name', 'TestLoss1', 'TestLoss1Relative', 'TestLoss2', 'TestLoss2Relative', 'TestLoss3', 'TestLoss3Relative'])
            # 4. 写入csv文件内容
            for i in range(len(result_list)):
                csv_writer.writerow(result_list[i])


if __name__ == '__main__':
    file_path = r'different_initial_interpolation_extrapolation_longtime/different_initial_all.txt'
    read_different_initial_txt(file_path, 9)

    file_path_A = r'different_A_initial_interpolation_extrapolation_longtime/different_A_initial_all.txt'
    read_different_A_initial_txt(file_path_A, 10)