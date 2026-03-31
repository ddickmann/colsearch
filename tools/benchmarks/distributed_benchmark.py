import mlx.core as mx
import mlx.nn as mnn
import time
import os

def main():
    # 1. Initialize Distributed Group
    # This automatically detects MPI world size and rank
    group = mx.distributed.init_dist(dist_backend="mpi")
    world_size = group.size()
    rank = group.rank()

    print(f"[Rank {rank}/{world_size}] 🟢 MLX Distributed Initialized")
    print(f"[Rank {rank}] 🖥️  Hostname: {os.uname().nodename}")

    # Synchronize all processes to ensure connection is solid
    mx.distributed.barrier(group)
    if rank == 0:
        print("\n✅ Connection Established! All peers recognized.")
        print("-" * 60)

    # 2. Bandwidth Test (All-Reduce)
    # create a large tensor (100MB)
    N = 1024 * 1024 * 25 # 25M floats = 100MB
    x = mx.random.normal((N,))
    
    # Warmup
    mx.distributed.all_sum(x, group=group)
    mx.eval(x)

    if rank == 0:
        print("🚀 Testing Bandwidth (100MB All-Reduce)...")
    
    start_time = time.time()
    for _ in range(10):
        # All machines sum their tensors together
        y = mx.distributed.all_sum(x, group=group)
        mx.eval(y)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 10
    
    # Each all-reduce sends 2 * size bytes roughly
    # 100MB * 2 = 200MB moved per step
    # GB/s = (0.2 GB) / time
    bandwidth_gbps = (0.2) / avg_time 

    if rank == 0:
        print(f"⏱️  Average Time: {avg_time:.4f} s")
        print(f"🌊 Effective Bandwidth: {bandwidth_gbps:.2f} GB/s")
        print("-" * 60)

        if bandwidth_gbps > 10.0:
            print("🚀 VERDICT: Ultra Fast (Thunderbolt?)")
        elif bandwidth_gbps > 0.5:
            print("🟢 VERDICT: Good (10GbE or fast Ethernet)")
        else:
            print("⚠️ VERDICT: Slow. Likely WiFi or standard Ethernet.")

    # 3. Simple Training Step Simulation
    if rank == 0:
        print("\n🛠️  Simulating Distributed Step...")

    # Everyone creates a small model
    model = mnn.Linear(1024, 1024)
    mx.eval(model.parameters())

    # Forward
    x_in = mx.random.normal((32, 1024)) # rank-local batch
    y_out = model(x_in)
    
    # Backward (fake gradient)
    grads = mx.random.normal((1024, 1024))
    
    # Sync Gradients (The core of distributed training)
    start_sync = time.time()
    grads_synced = mx.distributed.all_sum(grads, scale=1.0/world_size, group=group)
    mx.eval(grads_synced)
    
    if rank == 0:
        print(f"✅ Sync Complete in {time.time() - start_sync:.4f}s")
        print("\n🎉 Distributed Setup is READY.")

if __name__ == "__main__":
    main()
