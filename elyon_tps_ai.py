from planck_core import PlanckCoreAI
import torch
import time

class ElyonTPSAI(PlanckCoreAI):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)
        self.token_emission_rate = 0.00001

    def benchmark(self, input_tensor=None, trials=1000):
        if input_tensor is None:
            input_tensor = torch.randn(1, self.hidden.in_features)

        self.eval()
        inference_times = []

        with torch.no_grad():
            for _ in range(trials):
                start_time = time.time()
                _ = self(input_tensor)
                end_time = time.time()
                inference_times.append(end_time - start_time)

        avg_time = sum(inference_times) / trials
        tps = 1 / avg_time if avg_time > 0 else float('inf')
        earned_elyon = tps * self.token_emission_rate

        return {
            "Average Inference Time (s)": avg_time,
            "TPS": tps,
            "ElyonCoin Earned": earned_elyon
        }

if __name__ == "__main__":
    model = ElyonTPSAI(10, 20, 1)
    results = model.benchmark()
    print("\nElyon TPS Benchmark Results:")
    for key, value in results.items():
        print(f"{key}: {value:,.6f}")