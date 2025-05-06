# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
from pathlib import Path
from typing import Optional, Union

import docker

docker_client = docker.from_env()


def exists(image_name: str) -> bool:
    try:
        docker_client.images.get(image_name)
        return True
    except docker.errors.ImageNotFound:
        return False


def build(
    workdir: Path, dockerfile_name: str, image_full_name: str, logger: logging.Logger
):
    workdir = str(workdir)
    logger.info(
        f"Start building image `{image_full_name}`, working directory is `{workdir}`"
    )
    try:
        build_logs = docker_client.api.build(
            path=workdir,
            dockerfile=dockerfile_name,
            tag=image_full_name,
            rm=True,
            forcerm=True,
            decode=True,
            encoding="utf-8",
        )

        for log in build_logs:
            if "stream" in log:
                logger.info(log["stream"].strip())
            elif "error" in log:
                error_message = log["error"].strip()
                logger.error(f"Docker build error: {error_message}")
                raise RuntimeError(f"Docker build failed: {error_message}")
            elif "status" in log:
                logger.info(log["status"].strip())
            elif "aux" in log:
                logger.info(log["aux"].get("ID", "").strip())

        logger.info(f"image({workdir}) build success: {image_full_name}")
    except docker.errors.BuildError as e:
        logger.error(f"build error: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unknown build error occurred: {e}")
        raise e


def run(
    image_full_name: str,
    run_command: str,
    output_path: Optional[Path] = None,
    global_env: Optional[list[str]] = None,
    volumes: Optional[Union[dict[str, str], list[str]]] = None,
) -> str:
    container = docker_client.containers.run(
        image=image_full_name,
        command=run_command,
        remove=False,
        detach=True,
        stdout=True,
        stderr=True,
        environment=global_env,
        volumes=volumes,
    )

    output = ""
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for line in container.logs(stream=True, follow=True):
                line_decoded = line.decode("utf-8")
                f.write(line_decoded)
                output += line_decoded
    else:
        container.wait()
        output = container.logs().decode("utf-8")

    container.remove()

    return output
