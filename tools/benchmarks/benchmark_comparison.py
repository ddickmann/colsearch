import time
import torch
import torch.nn as nn
import mlx.core as mx
import mlx.nn as mnn
import mlx.optimizers as optim
import numpy as np

# --- CONFIGURATION ---
# 1B parameter approximation
# Hidden: 2048, Layers: 22 (approx), Vocab: 32k
BATCH_SIZE = 4
SEQ_LEN = 512
HIDDEN_SIZE = 2048
NUM_LAYERS = 22
VOCAB_SIZE = 32000
NUM_STEPS = 10  # Steps to average over
WARMUP_STEPS = 2

print("\n" + "="*60)
print(f"🚀  BENCHMARK: PyTorch (MPS) vs MLX (Metal) - 1B Model")
print(f"    Batch: {BATCH_SIZE} | Seq: {SEQ_LEN} | Hidden: {HIDDEN_SIZE} | Layers: {NUM_LAYERS}")
print("="*60 + "\n")

# --- PYTORCH IMPLEMENTATION ---
class TorchDummy1B(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False) 
            for _ in range(NUM_LAYERS * 3)
        ])
        self.head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return self.head(x)

def run_pytorch():
    if not torch.backends.mps.is_available():
        return 0, 0
    
    device = torch.device("mps")
    print(f"🔥  PyTorch (MPS): Initializing model...")
    model = TorchDummy1B().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    inputs = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).to(device)
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)

    # Warmup
    for _ in range(WARMUP_STEPS):
        output = model(inputs)
        loss = criterion(output.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.mps.synchronize()

    # Benchmark
    print(f"    Measuring {NUM_STEPS} steps...")
    start_time = time.time()
    for _ in range(NUM_STEPS):
        output = model(inputs)
        loss = criterion(output.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.mps.synchronize()
    
    total_time = time.time() - start_time
    avg_time = total_time / NUM_STEPS
    print(f"    PyTorch Avg Time: {avg_time:.4f} s/step")
    return avg_time

# --- MLX IMPLEMENTATION ---
class MLXDummy1B(mnn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            mnn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
            for _ in range(NUM_LAYERS * 3)
        ]
        self.head = mnn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return self.head(x)

def run_mlx():
    print(f"\n⚡  MLX (Metal): Initializing model...")
    model = MLXDummy1B()
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=1e-4)
    
    def loss_fn(model, x, y):
        logits = model(x)
        # Simple cross entropy equivalent
        # MLX cross_entropy expects logits, targets
        # Reshape for loss
        logits = logits.reshape(-1, VOCAB_SIZE)
        y = y.reshape(-1)
        return mnn.losses.cross_entropy(logits, y, reduction='mean')

    loss_and_grad_fn = mnn.value_and_grad(model, loss_fn)

    # Fake data
    inputs = mx.random.normal((BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE))
    targets = mx.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    # Warmup
    for _ in range(WARMUP_STEPS):
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    # Benchmark
    print(f"    Measuring {NUM_STEPS} steps...")
    start_time = time.time()
    for _ in range(NUM_STEPS):
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    
    total_time = time.time() - start_time
    avg_time = total_time / NUM_STEPS
    print(f"    MLX Avg Time:     {avg_time:.4f} s/step")
    return avg_time

# --- VISUALIZATION & METRICS ---
def estimate_flops(hidden, layers, seq_len, batch):
    # Very rough approx for Transformer training FLOPS
    # 6 * N * D^2 * L (N=params) - simplified
    # Lets assume ~1.1B params
    params = 1.1e9 
    # FLOPS per token per step ~ 6 * P
    flops_per_step = 6 * params * batch * seq_len
    return flops_per_step

def main():
    pt_time = run_pytorch()
    mlx_time = run_mlx()

    flops_per_step = estimate_flops(HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN, BATCH_SIZE)
    
    print("\n" + "="*60)
    print("🏆  RESULTS SUMMARY")
    print("="*60)
    
    headers = ["Framework", "Time/Step", "Speedup", "TFLOPS (est)", "Samples/Sec"]
    row_fmt = "{:<12} | {:<10} | {:<8} | {:<12} | {:<12}"
    
    print(row_fmt.format(*headers))
    print("-" * 65)

    # PyTorch Stats
    pt_tflops = (flops_per_step / pt_time) / 1e12 if pt_time > 0 else 0
    pt_samples = BATCH_SIZE / pt_time if pt_time > 0 else 0
    print(row_fmt.format("PyTorch", f"{pt_time:.3f}s", "1.00x", f"{pt_tflops:.2f}", f"{pt_samples:.1f}"))

    # MLX Stats
    mlx_tflops = (flops_per_step / mlx_time) / 1e12 if mlx_time > 0 else 0
    mlx_samples = BATCH_SIZE / mlx_time if mlx_time > 0 else 0
    speedup = pt_time / mlx_time if mlx_time > 0 else 0
    print(row_fmt.format("MLX", f"{mlx_time:.3f}s", f"{speedup:.2f}x", f"{mlx_tflops:.2f}", f"{mlx_samples:.1f}"))

    print("-" * 65)
    
    # Estimation
    print("\n🔮  ESTIMATED FINE-TUNING TIME (1 Epoch, 10k examples)")
    print(f"    Assuming batch size {BATCH_SIZE}:")
    
    steps_10k = 10000 / BATCH_SIZE
    pt_est = steps_10k * pt_time
    mlx_est = steps_10k * mlx_time
    
    def fmt_time(s):
        return time.strftime("%Hh %Mm %Ss", time.gmtime(s))

    print(f"    PyTorch: {fmt_time(pt_est)}")
    print(f"    MLX:     {fmt_time(mlx_est)}")
    
    if speedup > 1.1:
        print(f"\n✅  Recommendation: Use MLX. It is {speedup:.1f}x faster.")
    else:
        print(f"\n✅  Recommendation: Performance is similar. Use whichever you prefer.")

if __name__ == "__main__":
    main()
