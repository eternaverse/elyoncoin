import torch
import time
from planck_core import PlanckCoreAI
from tps_nft_generator import generate_tps_nft

input_size = 10
hidden_size = 20
output_size = 1
wallet_address = "0xElyonWalletAddressExample"

model = PlanckCoreAI(input_size, hidden_size, output_size)
model.eval()

sample_input = torch.randn(1, input_size)

num_trials = 1000
inference_times = []

with torch.no_grad():
    for _ in range(num_trials):
        start_time = time.time()
        _ = model(sample_input)
        end_time = time.time()
        inference_times.append(end_time - start_time)

avg_inference_time = sum(inference_times) / num_trials
transactions_per_second = 1 / avg_inference_time if avg_inference_time > 0 else float('inf')

elyon_per_tps = 0.00001
earned_elyon = transactions_per_second * elyon_per_tps

print(f"\nBenchmark Results:")
print(f"Average Inference Time: {avg_inference_time:.8f} seconds")
print(f"Transactions Per Second (TPS): {transactions_per_second:,.2f}")
print(f"ElyonCoin Earned: {earned_elyon:,.6f}")

generate_tps_nft(transactions_per_second, earned_elyon, wallet_address)