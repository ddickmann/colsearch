import torch
import torch.nn as nn
import time

# --- CONFIGURATION FOR 1B PARAMETER SIMULATION ---
# Approximating a TinyLlama-1.1B architecture
# Hidden size 2048, roughly 22 layers, vocabulary ~32k
BATCH_SIZE = 4       # Typical small batch size for local training
SEQ_LEN = 512        # Context length
HIDDEN_SIZE = 2048   
NUM_LAYERS = 22      

class Dummy1BModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a stack of layers to simulate the VRAM weight of a 1B model
        # We use a ModuleList of Linear layers to approximate the parameter count
        # 1.1B params roughly.
        self.layers = nn.ModuleList([
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False) 
            for _ in range(NUM_LAYERS * 3) # x3 approximates the Attention+FFN complexity
        ])
        self.head = nn.Linear(HIDDEN_SIZE, 32000, bias=False) # Vocab head

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x  # Residual connection to prevent vanishing gradients
        return self.head(x)

def run_benchmark():
    # 1. Check for Apple Silicon GPU (MPS)
    if not torch.backends.mps.is_available():
        print("❌ MPS (Metal) not found. This script is running on CPU (Slow).")
        device = torch.device("cpu")
    else:
        print("✅ Apple M1 Pro GPU detected (MPS backend).")
        device = torch.device("mps")

    print(f"🚀 Initializing ~1.1B Parameter Dummy Model...")
    
    # Instantiate model and move to GPU
    try:
        model = Dummy1BModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
    except Exception as e:
        print(f"❌ Failed to initialize model. Out of memory? Error: {e}")
        return

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"📊 Total Parameters: {param_count / 1e9:.2f} Billion")
    print(f"💾 Estimated Model VRAM usage (FP32): {param_count * 4 / 1024**3:.2f} GB (Weights only)")
    print("-" * 40)
    print("Beginning Training Loop Simulation (Forward + Backward + Optimizer)...")

    # Create dummy data
    inputs = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).to(device)
    targets = torch.randint(0, 32000, (BATCH_SIZE, SEQ_LEN)).to(device)

    # Warmup step (don't time this)
    print("Warmup step...", end="\r")
    output = model(inputs)
    loss = criterion(output.view(-1, 32000), targets.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Benchmark Loop
    num_steps = 5
    start_time = time.time()
    
    for i in range(num_steps):
        step_start = time.time()
        
        # 1. Forward
        output = model(inputs)
        loss = criterion(output.view(-1, 32000), targets.view(-1))
        
        # 2. Backward (Gradient calculation)
        loss.backward()
        
        # 3. Optimize (Weight update)
        optimizer.step()
        optimizer.zero_grad()
        
        # Force sync to get accurate timing on GPU
        if device.type == "mps":
            torch.mps.synchronize()
            
        print(f"Step {i+1}/{num_steps}: {time.time() - step_start:.4f} seconds")

    total_time = time.time() - start_time
    avg_time = total_time / num_steps

    print("-" * 40)
    print(f"✅ Benchmark Complete.")
    print(f"⏱️ Average Time Per Step: {avg_time:.4f}s")
    
    if avg_time < 1.0:
        print("🚀 VERDICT: Lightning fast. You can train much larger models.")
    elif avg_time < 5.0:
        print("🟢 VERDICT: Sufficient. Fine-tuning will be smooth.")
    else:
        print("⚠️ VERDICT: Slow, but functional. Training will take time.")

if __name__ == "__main__":
    run_benchmark()
