CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
	    --model "/root/GP/hstar/models/HVS-3B-sft-only" \
		--max_num_batched_tokens 40960 \
		--max-model-len 40960 \
	    --tensor-parallel-size 1 \
		--host 0.0.0.0 \
		--port 8000 \
		--trust_remote_code &
