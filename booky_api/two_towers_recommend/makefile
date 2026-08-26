.PHONY: test-api

test-api:
	@echo "Starting functions-framework..."
	@functions-framework --target=recommend --port=8080 & \
	SERVER_PID=$$!; \
	echo "Waiting for server to be ready..."; \
	until curl -s -o /dev/null "http://localhost:8080/?user_idx=1000&k=1"; do \
		sleep 0.5; \
	done; \
	echo "Server ready. Running test request..."; \
	curl -s "http://localhost:8080/?user_idx=1000&k=10" | python3 -m json.tool; \
	echo "Shutting down server (pid $$SERVER_PID)..."; \
	kill $$SERVER_PID