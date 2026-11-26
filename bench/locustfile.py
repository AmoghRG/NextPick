"""Locust load test for NextPick /event and /recommend endpoints."""

from __future__ import annotations

import random
import uuid

from locust import HttpUser, between, task


class ShopperUser(HttpUser):
    wait_time = between(0.05, 0.2)

    def on_start(self) -> None:
        self.session_id = f"load_{uuid.uuid4().hex[:10]}"
        self.sample_items = [
            {"article_id": "0706016001", "price": 0.033, "department_name": "Trouser", "product_group_name": "Garment Lower body"},
            {"article_id": "0706016002", "price": 0.033, "department_name": "Trouser", "product_group_name": "Garment Lower body"},
            {"article_id": "0372860001", "price": 0.013, "department_name": "Jersey Basic", "product_group_name": "Garment Upper body"},
        ]

    @task(1)
    def click_item(self) -> None:
        item = random.choice(self.sample_items)
        self.client.post(
            "/event",
            json={"session_id": self.session_id, **item},
        )

    @task(3)
    def get_recommendations(self) -> None:
        self.client.get(f"/recommend?session_id={self.session_id}&top_k=10")
