import httpx
import time

def run():
    client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0)
    
    print("Checking /health...")
    r = client.get("/health")
    print(f"Health: {r.status_code} - {r.json()}")
    
    # Get user ids
    from src.api.dependencies import load_resources, get_investigator
    load_resources()
    inv = get_investigator()
    batch_ids = inv.user_ids[:100]
    
    print("Scoring batch of 100 users for performance check...")
    t0 = time.time()
    r = client.post("/score", json={"user_ids": batch_ids})
    elapsed = time.time() - t0
    print(f"Batch score (100 users) took {elapsed:.2f}s")
    
    valid_id = batch_ids[0]
    t0 = time.time()
    r = client.get(f"/users/{valid_id}/risk")
    elapsed = time.time() - t0
    print(f"Single /risk took {elapsed:.4f}s")
    
    t0 = time.time()
    r = client.get(f"/users/{valid_id}/investigation")
    elapsed = time.time() - t0
    print(f"Single /investigation took {elapsed:.4f}s")
    
    r = client.get("/review-queue")
    print(f"/review-queue size: {len(r.json())}")
    print("DONE")

if __name__ == "__main__":
    run()
