import os
import os.path as osp
from prettytable import PrettyTable
import numpy as np


file_dir = './work_dir-PubMed-SNGNN' # CiteSeer Chameleon Texas Cora Wisconsin Cornell PubMed Squirrel Actor
results = {}
processed_results = {}
best_results = {}
best_results_mean_std = {}
# part_id
all_best_results = {}

for root, dirs, files in os.walk(file_dir):
    for file_name in files:
        file_path = osp.join(root, file_name)
        info = file_name.split('_')
        # print(info)
        part_id = info[-1].split('.')[0]
        res_key = '_'.join([info[i] for i in range(2, 12)])
        all_best_results.setdefault(info[0], {})[part_id] = []
        res_keys = info[0] + '_' + res_key
        results.setdefault(res_keys, {})[info[1]] = []
        processed_results.setdefault(res_keys, {})[info[1]] = ''
        best_results.setdefault(info[0], {})[info[1]] = 0.0
        best_results_mean_std.setdefault(info[0], {})[info[1]] = 0.0
    for file_name in files:
        file_path = osp.join(root, file_name)
        info = file_name.split('_')
        part_id = info[-1].split('.')[0]
        res_key = '_'.join([info[i] for i in range(2, 12)])
        res_keys = info[0] + '_' + res_key
        with open(file_path, 'r') as f:
            lines = f.readlines()
            test_acc = lines[-1].strip()[-6:]
            try:
                all_best_results.setdefault(info[0], {})[part_id].append(float(test_acc))
            except:
                print(file_path)
            # print(test_acc)
            try:
                results.setdefault(res_keys, {})[info[1]].append(float(test_acc))
            except:
                print(file_path)


def mean_std(arr):
    arr_mean = np.mean(arr)*100
    arr_std = np.std(arr, ddof=1) * 100
    return '{:.2f}±{:.2f}'.format(arr_mean, arr_std)


# def mean_arr(arr):
#     arr_mean = np.mean(arr)*100
#     return arr_mean
#
# print(results)
# print(processed_results)
# print(results[list(results.keys())[0]].keys())
# print(processed_results[list(results.keys())[0]].keys())
# model_lst = sorted(list(results.keys()))
# dataset_lst = list(results[list(results.keys())[0]].keys())
# for key_0 in results.keys():
#     # for i in range(len(results.keys())):
#     for key_1 in results[list(results.keys())[0]].keys():
#         processed_results[key_0][key_1] = mean_std(results[key_0][key_1])


for i in range(len(results.keys())):
    key_0 = list(results.keys())[i]
    model_name = key_0.split('_')[0]
    for key_1 in results[list(results.keys())[i]].keys():
        processed_results[key_0][key_1] = mean_std(results[key_0][key_1])
        model_acc = float(mean_std(results[key_0][key_1]).split('±')[0])
        if model_acc > best_results[model_name][key_1]:
            best_results[model_name][key_1] = model_acc
            best_results_mean_std[model_name][key_1] = mean_std(results[key_0][key_1])

# print(best_results_mean_std)

# print(processed_results)
table = PrettyTable()
field_name = ['model'] + sorted(list(processed_results[list(processed_results.keys())[0]].keys()))
rows = []
for key in sorted(list(processed_results.keys())):
    tmp_row = [key]
    for new_key in field_name[1:]:
        val = processed_results[key].get(new_key)
        tmp_row.append(val)
        # tmp_row.append(processed_results[key][new_key])
    rows.append(tmp_row)


table.field_names = field_name
for i in range(len(rows)):
    table.add_row(rows[i])

print(table)

table2 = PrettyTable()
field_name = ['model'] + sorted(list(best_results_mean_std[list(best_results_mean_std.keys())[0]].keys()))
rows2 = []
for key in sorted(list(best_results_mean_std.keys())):
    tmp_row = [key]
    for new_key in field_name[1:]:
        val = best_results_mean_std[key].get(new_key)
        tmp_row.append(val)
        # tmp_row.append(processed_results[key][new_key])
    rows2.append(tmp_row)

table2.field_names = field_name
for i in range(len(rows2)):
    table2.add_row(rows2[i])

print(table2)


# print(all_best_results)
all_best = {}
for key_0 in all_best_results.keys():
    # all_best[key_0] = []
    all_best.setdefault(key_0, [])
    for key_1 in all_best_results[key_0].keys():
        all_best[key_0].append(max(all_best_results[key_0][key_1]))
# print(all_best)
for k in all_best:
    print('{}'.format(mean_std(all_best[k])))