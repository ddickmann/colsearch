# Distributed Training Setup Guide for Mac (MLX)

This guide walks you through connecting multiple Macs to train large AI models together using Apple's MLX framework.

## 1. Physical Connection (Thunderbolt Bridge)
**Recommended**: This is the fastest method (up to 40Gbps).

1.  **Connect the Cable**: Plug a Thunderbolt cable directly connecting Mac A and Mac B.
2.  **Verify Connection**:
    *   Open **System Settings** > **Network**.
    *   Look for **Thunderbolt Bridge** (it should have a yellow/green dot).
    *   Note the IP address on both machines (usually `169.254.x.x`).
    *   *Tip: If you don't see an IP, manually set "Configure IPv4" to "Manually" and assign `10.0.0.1` and `10.0.0.2` with subnet mask `255.255.255.0` to ensure a stable link.*

## 2. Software Prerequisites (On ALL Machines)
You must install these on **every** Mac you plan to use.

### Install OpenMPI
OpenMPI handles the communication between the machines.
```bash
brew install open-mpi
```

### Install MLX Dependencies
Ensure your Python environment has the latest MLX.
```bash
# Activate your environment
source venv/bin/activate

# Install/Update MLX
pip install mlx mlx-lm
```

## 3. SSH Configuration (Password-less Access)
for the "Master" machine to launch processes on the "Worker" machine, it needs SSH access without typing a password every time.

1.  **Generate SSH Key** (On Main Node):
    ```bash
    ssh-keygen -t ed25519
    # Press Enter through all prompts (no passphrase for automation)
    ```

2.  **Copy Key to Worker** (On Main Node):
    Replace `user@worker-ip` with the actual username and IP of your second Mac.
    ```bash
    ssh-copy-id user@169.254.123.456
    ```

3.  **Verify**:
    ```bash
    ssh user@169.254.123.456
    # You should log in immediately without a password prompt.
    ```

## 4. running the Benchmark
Once connected, navigating to the project folder on **both** machines.

**Command (Run on Main Node only):**
```bash
mpirun -np 2 \
    -host localhost,169.254.123.456 \
    python distributed_benchmark.py
```
*   `-np 2`: Number of processes (2 Macs = 2).
*   `-host`: Comma-separated list of IPs.
