import sys
import os
import time
import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, os.getcwd())

from moai.n2he import N2HEContext, KeyGenerator, Encryptor, Decryptor, Evaluator

def benchmark_fhe():
    print("=" * 60)
    print("FHE LATENCY BENCHMARK (High-Fidelity Simulation)")
    print("=" * 60)
    
    # Setup
    ctx = N2HEContext()
    keygen = KeyGenerator(ctx)
    encryptor = Encryptor(ctx, keygen.public_key())
    decryptor = Decryptor(ctx, keygen.secret_key())
    evaluator = Evaluator(ctx)
    
    # Data: 224x224x3 Image (Standard ViT input)
    frame = np.random.rand(224, 224, 3).astype(np.float32)
    print(f"Input Frame: {frame.shape} ({frame.nbytes/1024:.1f} KB)")
    
    results = []
    
    # 1. Encryption Latency
    start = time.time()
    ct_frame = encryptor.encrypt(frame)
    enc_time = (time.time() - start) * 1000
    print(f"Encryption Time: {enc_time:.1f} ms")
    print(f"Ciphertext Size: {ct_frame.size_bytes()/1024/1024:.2f} MB (Expansion: {ct_frame.size_bytes()/frame.nbytes:.1f}x)")
    
    results.append({"Operation": "Encryption", "Latency (ms)": enc_time, "Size (MB)": ct_frame.size_bytes()/1024/1024})
    
    # 2. Transmission Latency (Simulated Wi-Fi 6E)
    # Wi-Fi 6E Real-world ~1 Gbps = 125 MB/s
    # Upload time = Size / Speed
    upload_speed_mbps = 125.0 
    tx_time = (ct_frame.size_bytes() / (1024*1024) / upload_speed_mbps) * 1000
    print(f"Transmission Time (Simulated 1Gbps): {tx_time:.1f} ms")
    results.append({"Operation": "Transmission", "Latency (ms)": tx_time, "Size (MB)": ct_frame.size_bytes()/1024/1024})
    
    # 3. Encrypted Compute (MOAI Layer)
    # Simulate a Transformer Block: Attention + MLP
    # Attention: Add + Mult + Rotate
    # MLP: Mult + Add
    
    print("\nSimulating MOAI Transformer Block (Encrypted)...")
    compute_start = time.time()
    
    # Self-Attention (Simplified)
    # Q * K^T (Mult)
    ct_attn = evaluator.multiply(ct_frame, ct_frame) 
    # Softmax (Approx) -> Rotate + Add
    ct_attn = evaluator.rotate(ct_attn, 1)
    ct_attn = evaluator.add(ct_attn, ct_attn)
    
    # MLP
    # Linear (Mult)
    ct_mlp = evaluator.multiply(ct_attn, ct_attn)
    # Add residual
    ct_out = evaluator.add(ct_mlp, ct_frame)
    
    compute_time = (time.time() - compute_start) * 1000
    print(f"Encrypted Compute Time: {compute_time:.1f} ms")
    results.append({"Operation": "Encrypted Compute", "Latency (ms)": compute_time, "Size (MB)": ct_out.size_bytes()/1024/1024})
    
    # 4. Decryption Latency
    start = time.time()
    _ = decryptor.decrypt(ct_out)
    dec_time = (time.time() - start) * 1000
    print(f"Decryption Time: {dec_time:.1f} ms")
    results.append({"Operation": "Decryption", "Latency (ms)": dec_time, "Size (MB)": 0})
    
    # Total Round Trip
    total_rtt = enc_time + tx_time + compute_time + dec_time
    print(f"\nTotal Round Trip Time: {total_rtt:.1f} ms ({total_rtt/1000:.2f} s)")
    
    # Comparison with Native
    print("\n" + "-" * 60)
    print("COMPARISON: REFLEX (Native) vs CORTEX (FHE)")
    print("-" * 60)
    
    # Native Compute (PyTorch CPU estimate for AGX Orin)
    # ResNet/ViT inference ~10-30ms
    native_compute = 20.0 
    native_tx = (frame.nbytes / (1024*1024) / upload_speed_mbps) * 1000 # Unencrypted upload
    
    print(f"Reflex (Native) Latency: ~{native_compute:.1f} ms")
    print(f"Cortex (FHE) Latency:    ~{total_rtt:.1f} ms")
    print(f"Slowdown Factor:         {total_rtt/native_compute:.1f}x")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("fhe_benchmark_results.csv", index=False)
    print("\nResults saved to fhe_benchmark_results.csv")

if __name__ == "__main__":
    benchmark_fhe()
