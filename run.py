import subprocess

# Step 1: Build and run Docker container
docker_build_cmd = [
    "docker", "build", "-t", "my_dashboard_app", "."
]
docker_run_cmd = [
    "docker", "run", "-d", "-p", "8501:8501", "--name", "dashboard_container", "my_dashboard_app"
]

# Step 2: Run Python dashboard (assumes dashboard.py is the entry point)
# This should be part of your Dockerfile, but for local run:
dashboard_cmd = [
    "python", "dashboard.py"
]

def run_docker():
    subprocess.run(docker_build_cmd, check=True)
    subprocess.run(docker_run_cmd, check=True)

def run_dashboard():
    subprocess.run(dashboard_cmd, check=True)

if __name__ == "__main__":
    # Uncomment one of the following depending on your use case:
    # To run in Docker:
    # run_docker()
    # To run locally:
    # run_dashboard()
    pass