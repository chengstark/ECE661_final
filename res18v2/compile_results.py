import os
import numpy as np
import pandas as pd

def parse_results(file_lines, eval, f_name=None):
    test_accs = []
    configs = None

    for line in file_lines:
        if ' | ' in line and ';' in line:
            tets_acc = line.split(';')[1].split('Acc')[1]
            test_accs.append(float(tets_acc))

        if 'valwtest' in line:
            configs = line.split('valwtest')[1].split('.pth')[0].split('_')[1:]
            configs[0] = ([int(x) for x in configs[0].split('*')][0]+1) * 100

        if eval == 'semi':
            if 'label_percentage' in line:
                label_perc = float(line.split('label_percentage=')[1].split(',')[0])
                configs.append(label_perc)

    max_test_acc = max(test_accs)
    if eval == 'linear': configs.append(1.0)

    return configs, max_test_acc

result_file = open('results_compiled.txt', 'w')

out_files = [f for f in os.listdir() if f.endswith('.out')]
results = []
results_for_csv = []
for f in out_files:
    file = open(f, 'r')
    file_contents = file.read()
    file_lines = file_contents.split('\n')

    if 'Acc' in file_contents and 'BASELINEBASELINE' not in file_contents:
        if 'label_percentage' in file_contents:
            configs, max_test_acc = parse_results(file_lines, eval='semi', f_name=f)
            result_line = f'Semi-sup | Epoch {configs[0]}/{configs[1]}, pretrain_bs {configs[2]}, Temp {configs[3]}, LR {configs[4]}, label_perc {configs[5]}, test_acc {max_test_acc}'
            results_for_csv.append(configs+[max_test_acc]+['semi'])
        else:
            configs, max_test_acc = parse_results(file_lines, eval='linear', f_name=f)
            result_line = f'Linear-- | Epoch {configs[0]}/{configs[1]}, pretrain_bs {configs[2]}, Temp {configs[3]}, LR {configs[4]}, label_perc {configs[5]}, test_acc {max_test_acc}'
            results_for_csv.append(configs+[max_test_acc]+['linear'])

        results.append(result_line)

results = sorted(results)
for line in results:
    result_file.write(line+'\n')
result_file.close()


df = pd.DataFrame(results_for_csv, columns = ['epoch', 'epoch_total', 'pretrain_bs', 'temp', 'lr', 'label_perc', 'test_acc', 'eval'])
df.to_csv('results_compiled.csv', index=False)
