FROM ollama/ollama:latest

ENV MODEL=gemma3:latest

RUN ollama pull $MODEL

ENTRYPOINT ["ollama", "serve"]
