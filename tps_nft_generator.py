import uuid
import json
import os
from datetime import datetime

def generate_tps_nft(tps, earned_elyon, wallet_address, output_dir="tps_nfts"):
    os.makedirs(output_dir, exist_ok=True)

    nft_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    metadata = {
        "name": f"TPS Proof NFT #{nft_id[:8]}",
        "description": "Proof of AI Benchmark Performance on PlanckCoreAI",
        "attributes": [
            {"trait_type": "TPS", "value": round(tps, 2)},
            {"trait_type": "ElyonCoin Earned", "value": round(earned_elyon, 6)},
            {"trait_type": "Benchmark Time", "value": timestamp},
            {"trait_type": "Wallet Address", "value": wallet_address}
        ],
        "external_url": "https://elyon.ai/tps",
        "image": "https://elyon.ai/nft-placeholder.png",
        "id": nft_id
    }

    filename = f"{output_dir}/tps_nft_{nft_id}.json"
    with open(filename, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ TPS NFT metadata saved: {filename}")
    return filename