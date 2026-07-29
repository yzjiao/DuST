FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Pre-install common packages that solutions might import
RUN pip install --no-cache-dir \
    numpy scipy sympy sortedcontainers networkx

WORKDIR /sandbox
COPY server.py .

ENV PORT=8080
EXPOSE 8080

CMD ["python", "server.py"]
