import subprocess
from tqdm import tqdm
import os
import argparse
import asyncio
import numpy as np
import gc  # Import garbage collector module

from fractions import Fraction
from itertools import product

# Function to run a single experiment
async def run_experiment(args, gpu):
    N_agent_, N_prey, percentage, total_steps_per_simulation, sensor_range_, sensing_range_, exp_id, folder, save_last_step = args
    if gpu:
        process = await asyncio.create_subprocess_exec(
            'python', 'main.py',
            '--N', str(N_agent_),
            '--N_preys', str(N_prey),
            '--no_sensor', str(percentage),
            '--steps', str(total_steps_per_simulation),
            '--sensor_range', str(sensor_range_),
            '--sensing_range', str(sensing_range_),
            '--experiment_id', exp_id,
            '--save',
            '--no_progress_bar',
            '--gpu',
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    else:
        if not save_last_step:
            process = await asyncio.create_subprocess_exec(
                'python', 'main.py',
                '--N', str(N_agent_),
                '--N_preys', str(N_prey),
                '--no_sensor', str(percentage),
                '--steps', str(total_steps_per_simulation),
                '--sensor_range', str(sensor_range_),
                '--sensing_range', str(sensing_range_),
                '--experiment_id', exp_id,
                '--folder', folder,
                '--save',
                # '--pdm',
                # '--pdm_prey', #comment here!
                # '--Dp', str(4),
                '--no_progress_bar',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        else:
            process = await asyncio.create_subprocess_exec(
                'python', 'main.py',
                '--N', str(N_agent_),
                '--N_preys', str(N_prey),
                '--no_sensor', str(percentage),
                '--steps', str(total_steps_per_simulation),
                '--sensor_range', str(sensor_range_),
                '--sensing_range', str(sensing_range_),
                '--experiment_id', exp_id,
                '--folder', folder,
               '--pdm',
            #    '--pdm_prey', #comment here!
                # '--Dp', 5,
                '--no_progress_bar',
                '--save_last_step',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
    await process.communicate()
    
    # Clean up memory
    del process
    gc.collect()

# Function to run a batch of experiments
async def run_batch(batch, gpu):
    tasks = []
    for args in batch:
        tasks.append(run_experiment(args, gpu))
    await asyncio.gather(*tasks)
    
    # Clean up memory
    del tasks
    gc.collect()

# Function to run all experiments
def run_experiments(gpu: bool = False, num_exps: int = 1, folder: str = 'results', save_last_step: bool = False):
    # Experiment IDs
    experiment_id_ = [str(i) for i in range(1, num_exps + 1)]
    
    # Parameters
    folders = [folder]
    N_preys = [0]
    N_agent = [100]
    # N_preys = [100, 50]
    # N_agent = [100, 50]
    sensing_range = [3]
    sensor_range = [3]
    # sensing_range = [2, 3, 4]
    # sensor_range =  [2, 3, 4]
    save_last_steps = [save_last_step]
    # percentage_no_sensor = np.linspace(0, 0.99, 10)
    # percentage_no_sensor = np.round(percentage_no_sensor, 2)
    percentage_no_sensor = [0]
    total_steps_per_simulation = 30000
    
    specific_ratios = [Fraction(1, 1), Fraction(2, 1), Fraction(3, 2), Fraction(1, 2), Fraction(2, 3)]

    allowed_ratios = set([Fraction(sensor, sensing).limit_denominator() for sensing, sensor in product(sensing_range, sensor_range) if Fraction(sensor, sensing).limit_denominator() in specific_ratios])
    allowed_reversed_ratios = set([Fraction(sensor, sensing).limit_denominator() for sensing, sensor in product(sensor_range, sensing_range) if Fraction(sensor, sensing).limit_denominator() in specific_ratios])
    allowed_ratios = allowed_ratios.union(allowed_reversed_ratios)
    
    valid_combinations = []
    for sr in sensor_range:
        for sensing_r in sensing_range:
            ratio = Fraction(sr, sensing_r)
            reversed_ratio = (sensing_r, sr)
            if ratio in allowed_ratios or reversed_ratio in allowed_ratios:
                valid_combinations.append((sr, sensing_r))
    
    total_iterations = (len(N_preys) * len(percentage_no_sensor) * total_steps_per_simulation *
                        len(N_agent) * len(valid_combinations) *
                        len(experiment_id_) * len(folder))

    if gpu:
        assert "CUDA_PATH" in os.environ, "CUDA not found, GPU is not available"

    batches = []
    for folder in folders:
        for exp_id in experiment_id_:
            for N_prey in N_preys:
                for percentage in percentage_no_sensor:
                    for N_agent_ in N_agent:
                        for (sensor_range_, sensing_range_) in valid_combinations:
                            for _ in save_last_steps:
                                batches.append((N_agent_, N_prey, percentage, total_steps_per_simulation, sensor_range_, sensing_range_, exp_id, folder, save_last_step))
    
    batch_size = 10
    num_batches = len(batches)
    print(f'Total files: {num_batches}')
    with tqdm(total=num_batches, desc='Experiment simulations progress') as pbar:
        for i in range(0, num_batches, batch_size):
            batch = batches[i:i + batch_size]
            asyncio.run(run_batch(batch, gpu))
            pbar.update(len(batch))
    
    # Final clean up
    del batches
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the experiments')
    parser.add_argument('--gpu', action='store_true', help='Run the simulations on the GPU')
    parser.add_argument('--folder', type=str, default='results', help='Folder to save the results')
    parser.add_argument('--num_exps', type=int, default=1, help='Number of experiments to run')
    parser.add_argument('--save_last_step', action='store_true', help='Save the last step')
    args = parser.parse_args()
    kwargs = vars(args)
    run_experiments(**kwargs)