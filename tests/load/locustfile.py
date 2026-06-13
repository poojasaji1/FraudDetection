import random
from datetime import datetime, timezone

from locust import HttpUser, task, between


class FraudUser(HttpUser):
    wait_time = between(0.01, 0.1)

    @task
    def predict(self):
        payload = {
            "user_id":     str(random.randint(1000, 9999)),
            "amount":      round(random.uniform(1.0, 500.0), 2),
            "merchant_id": str(random.randint(100, 999)),
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "location": {
                "lat": round(random.uniform(25.0, 48.0), 4),
                "lng": round(random.uniform(-122.0, -70.0), 4),
            },
        }
        with self.client.post("/v1/predict", json=payload, catch_response=True) as r:
            elapsed_ms = r.elapsed.total_seconds() * 1000
            if elapsed_ms > 100:
                r.failure(f"Too slow: {elapsed_ms:.1f}ms")
            elif r.status_code != 200:
                r.failure(f"Status {r.status_code}")
            else:
                r.success()
