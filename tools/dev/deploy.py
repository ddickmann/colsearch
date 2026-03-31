import yaml
import subprocess
import argparse
import os
import sys

CONFIG_FILE = "cluster_config.yaml"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Config file {CONFIG_FILE} not found!")
        sys.exit(1)
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def run_cmd(cmd, description=None):
    if description:
        print(f"➜ {description}...")
    try:
        subprocess.check_call(cmd, shell=True)
        print("  ✅ Done.")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error: {e}")
        sys.exit(1)

def sync_code(config, target="master"):
    """Syncs local code to the remote node(s)."""
    
    # Identify targets
    nodes = []
    if target == "master" or target == "all":
        nodes.append(config["master_node"])
    if target == "workers" or target == "all":
        nodes.extend(config["worker_nodes"])
        
    excludes = " ".join([f"--exclude='{x}'" for x in config.get("sync_exclude", [])])
    
    print(f"\n🔄 Syncing code to {len(nodes)} node(s)...")
    
    for node in nodes:
        user = node["user"]
        host = node["host"]
        dest_path = node["path"]
        
        # Ensure remote directory exists
        run_cmd(f"ssh {user}@{host} 'mkdir -p {dest_path}'", f"Creating dir on {host}")
        
        # Rsync
        # -a: archive mode, -v: verbose, -z: compress
        cmd = f"rsync -avz {excludes} ./ {user}@{host}:{dest_path}/"
        run_cmd(cmd, f"Syncing to {host}")

def setup_remote(config):
    """Installs dependencies on remote nodes."""
    nodes = [config["master_node"]] + config["worker_nodes"]
    
    print("\n🛠️  Setting up remote environments...")
    setup_cmds = [
        "pip install --upgrade pip",
        "pip install mlx mlx-lm pyyaml"
    ]
    
    for node in nodes:
        user = node["user"]
        host = node["host"]
        print(f"  Target: {host}")
        
        for cmd in setup_cmds:
            remote_cmd = f"ssh {user}@{host} '{cmd}'"
            run_cmd(remote_cmd, f"Running {cmd} on {host}")

def run_training(config, script="distributed_benchmark.py"):
    """Launches the distributed job on the Master node."""
    master = config["master_node"]
    workers = config["worker_nodes"]
    
    # Construct comma-separated host string for MPI
    # Localhost (master) + Worker IPs (mapped from master's perspective)
    # NOTE: This assumes the Master can reach Workers via IPs in config, 
    # or they share the same Thunderbolt network.
    
    # For Thunderbolt mesh, Master sees itself as localhost, and worker as config IP
    hosts = ["localhost"] + [w["host"] for w in workers]
    host_str = ",".join(hosts)
    np = len(hosts)
    
    user = master["user"]
    host = master["host"]
    path = master["path"]
    
    print(f"\n🚀 Launching MPI Job on Master Node ({host})...")
    print(f"   Processes: {np}")
    print(f"   Hosts: {host_str}")
    print(f"   Script: {script}")
    print("-" * 60)
    
    # The command to execute on the Master node via SSH
    # We use -u for unbuffered python output to see it in real-time
    remote_command = (
        f"cd {path} && "
        f"mpirun -np {np} -host {host_str} python -u {script}"
    )
    
    # SSH and stream output
    ssh_cmd = f"ssh {user}@{host} '{remote_command}'"
    
    # We use subprocess.call to pipe stdout/stderr directly to current terminal
    subprocess.call(ssh_cmd, shell=True)

def main():
    parser = argparse.ArgumentParser(description="Omni-Cluster Deployment Tool")
    parser.add_argument("--sync", action="store_true", help="Sync code to all nodes")
    parser.add_argument("--setup", action="store_true", help="Install dependencies on all nodes")
    parser.add_argument("--run", type=str, default=None, help="Run a specific script (e.g. distributed_benchmark.py)")
    parser.add_argument("--train", action="store_true", help="Run the default training workflow")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.setup:
        setup_remote(config)
        
    if args.sync or args.run or args.train:
        sync_code(config, target="all")
        
    if args.run:
        run_training(config, script=args.run)
    elif args.train:
        run_training(config, script="distributed_benchmark.py") # Default for now

    if not (args.sync or args.setup or args.run or args.train):
        parser.print_help()

if __name__ == "__main__":
    main()
