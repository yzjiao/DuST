#!/usr/bin/env python3
"""
Launch code execution sandbox containers for solution verification.

The sandbox provides an HTTP API for executing Python code against test cases.
This script starts N sandbox instances using Docker for parallel verification.

Usage:
    # Start 64 sandbox instances on ports 8080-8143
    python data_generation/0_launch_sandbox.py --num-instances 64

    # Start with custom config
    python data_generation/0_launch_sandbox.py --config configs/paths.yaml

Prerequisites:
    - Docker installed and running
    - Sandbox Docker image built/pulled (see configs/paths.yaml for image name)

The sandbox image must expose an HTTP server with the following API:
    POST /execute
    Body: {"code": str, "input": str, "timeout": int, "memory_limit_mb": int}
    Response: {"stdout": str, "stderr": str, "exit_code": int, "timed_out": bool}
"""

import argparse
import subprocess
import sys
import time
import yaml


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def start_sandbox(image, port, memory_limit_mb, container_name):
    """Start a single sandbox container."""
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "-p", f"{port}:{port}",
        "--memory", f"{memory_limit_mb}m",
        "--cpus", "2",
        "-e", f"PORT={port}",
        image,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Failed to start {container_name}: {result.stderr.strip()}")
        return False
    return True


def wait_for_health(host, port, timeout=30):
    """Wait for sandbox to respond on health endpoint."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            url = f"http://{host}:{port}/health"
            urllib.request.urlopen(url, timeout=2)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Launch sandbox containers")
    parser.add_argument("--config", default="configs/paths.yaml", help="Path config file")
    parser.add_argument("--num-instances", type=int, help="Override number of instances")
    parser.add_argument("--base-port", type=int, default=8080, help="Starting port number")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    config = load_config(args.config)
    sandbox_cfg = config["sandbox"]

    image = sandbox_cfg["docker_image"]
    num_instances = args.num_instances or sandbox_cfg["num_instances"]
    memory_limit = sandbox_cfg["memory_limit_mb"]
    base_port = args.base_port

    print(f"Launching {num_instances} sandbox containers")
    print(f"  Image: {image}")
    print(f"  Ports: {base_port}-{base_port + num_instances - 1}")
    print(f"  Memory limit: {memory_limit}MB per container")

    if args.dry_run:
        print("\n[DRY RUN] Would start containers with:")
        for i in range(min(3, num_instances)):
            port = base_port + i
            print(f"  docker run -d --name sandbox_{i} -p {port}:{port} {image}")
        if num_instances > 3:
            print(f"  ... ({num_instances - 3} more)")
        return

    started = 0
    for i in range(num_instances):
        port = base_port + i
        name = f"sandbox_{i}"

        # Remove existing container if any
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
        )

        if start_sandbox(image, port, memory_limit, name):
            started += 1
            if i < 5 or i == num_instances - 1:
                print(f"  [{i+1}/{num_instances}] Started {name} on port {port}")
        else:
            print(f"  [{i+1}/{num_instances}] FAILED {name}")

    print(f"\n{started}/{num_instances} sandbox containers started.")

    # Write sandbox endpoints to file for use by generation script
    endpoints_file = "sandbox_endpoints.txt"
    with open(endpoints_file, "w") as f:
        for i in range(num_instances):
            f.write(f"http://localhost:{base_port + i}\n")
    print(f"Endpoints written to {endpoints_file}")

    # Health check
    print("Running health checks...")
    healthy = 0
    for i in range(num_instances):
        port = base_port + i
        if wait_for_health("localhost", port, timeout=10):
            healthy += 1

    print(f"Health check: {healthy}/{started} containers responding")
    if healthy < started:
        print("WARNING: Some containers failed health check. Check docker logs.")


if __name__ == "__main__":
    main()
