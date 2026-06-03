import os
import shutil
import subprocess


def start_server():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    if os.name == "nt":
        if shutil.which("docker") is None:
            raise RuntimeError(
                "Docker is required to run the Overcooked demo. "
                "Install and start Docker Desktop, then run overcooked-demo-up again."
            )
        os.environ.setdefault("BUILD_ENV", "development")
        subprocess.call(["docker", "compose", "up", "--build"])
    else:
        subprocess.call(["sh", "./up.sh"])


def move_agent():
    from overcooked_demo.server.move_agents import main

    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.join(dir_path, "server"))
    main()
