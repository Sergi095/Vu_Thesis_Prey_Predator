import os
import subprocess
import sys

def clone_or_update_repo(repo_url, repo_dir):
    if os.path.exists(repo_dir):
        print(f"Repository already exists at {repo_dir}, pulling latest changes...")
        subprocess.run(['git', '-C', repo_dir, 'pull'], check=True)
    else:
        print(f"Cloning repository from {repo_url} to {repo_dir}...")
        subprocess.run(['git', 'clone', repo_url, repo_dir], check=True)

def main():
    repo_url = 'https://github.com/utiasDSL/gym-pybullet-drones.git'
    repo_dir = 'gym-pybullet-drones'

    clone_or_update_repo(repo_url, repo_dir)

    # Now install the package
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', repo_dir], check=True)

if __name__ == '__main__':
    main()
