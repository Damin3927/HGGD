.PHONY: build
build:
	docker build -t hggd-grasp \
		--platform linux/amd64 \
		.

.PHONY: run-gpu
run-gpu:
	docker run --rm -it \
		--gpus all \
		-v $(shell pwd):/app \
		hggd-grasp

.PHONY: run-cpu
run-cpu:
	docker run --rm -it \
		--platform linux/amd64 \
		-v $(shell pwd):/app \
		hggd-grasp
