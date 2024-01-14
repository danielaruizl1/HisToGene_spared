import json
import subprocess
import argparse
import os

# Get parsed the path of the config file
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_visium.json',    help='Path to the .json file with the configs of the dataset.')
args = parser.parse_args()

# Read the configs
with open(args.config, 'rb') as f:
    config_params = json.load(f)

# Create the command to run. If sota key is "None" call main.py else call main_sota.py
command_list = ['python', 'main.py']

for key, val in config_params.items():
    command_list.append(f'--{key}')
    command_list.append(f'{val}')

# Call subprocess
subprocess.call(command_list)