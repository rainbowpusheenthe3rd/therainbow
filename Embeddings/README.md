Qwen offer their embedding model via Docker / hf as of writing (23/09/2025)

# To run via a GPU
docker run --gpus all -p 8080:80 -v hf_cache:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.7.2 --model-id Qwen/Qwen3-Embedding-8B --dtype float16

# Tu run via CPU
docker run -p 8080:80 -v hf_cache:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.7.2 --model-id Qwen/Qwen3-Embedding-8B --dtype float16

Once this is done, the container should be available on port 8080.
curl http://localhost:8080/embed \
    -X POST \
    -d '{"inputs": ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of China?", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: Explain gravity"]}' \
    -H "Content-Type: application/json"