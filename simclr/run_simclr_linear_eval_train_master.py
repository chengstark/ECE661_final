import os
import subprocess


pretrained_model_folder = '/home/users/zg78/ece661_final/simclr/models/simclr_pretrained/'
pretrained_model_paths = [pretrained_model_folder+p for p in os.listdir(pretrained_model_folder)]
print('Running', pretrained_model_paths)


for p in pretrained_model_paths:
    subprocess.run(f'sbatch run_simclr_linear_eval_train_slave.sh {p}', shell=True)