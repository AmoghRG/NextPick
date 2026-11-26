.PHONY: test up demo api web bench

test:
	PYTHONPATH=. pytest -v

up:
	docker compose up -d

demo:
	PYTHONPATH=. python scripts/generate_demo_artifacts.py
	docker compose up -d
	PYTHONPATH=. python src/index/qdrant_loader.py

api:
	NEXTPICK_ALLOW_INMEMORY_REDIS=1 PYTHONPATH=. uvicorn src.serving.main:app --host 0.0.0.0 --port 8000

web:
	cd web && npm install && npm run dev

bench:
	locust -f bench/locustfile.py --headless -u 20 -r 5 -t 15s --host http://127.0.0.1:8000 --only-summary
