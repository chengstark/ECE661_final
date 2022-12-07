import os
import subprocess

# Note: ran 'valwtest_5' and 'valwtest_4'
# To cancel all user jobs: # squeue -u $USER | awk '{print $1}' | tail -n+2 | xargs scancel


pretrained_model_folder = '/home/users/zg78/ece661_final/simclr/models/simclr_pretrained/'
pretrained_model_paths = [pretrained_model_folder+p for p in os.listdir(pretrained_model_folder) if ('valwtest_3' in p or 'valwtest_4' in p)]
print('Running', pretrained_model_paths, len(pretrained_model_paths))

for label_perc in [0.1, 0.01]:
    for p in pretrained_model_paths:
        subprocess.run(f'sbatch run_simclr_semi_sup_train_slave.sh {p} {label_perc}', shell=True)



# for id in range(2729163, 2729182+1):
#     subprocess.run(f'scancel {id}', shell=True)