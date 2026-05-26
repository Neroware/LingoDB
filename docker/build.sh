#!/usr/bin/env bash

set -e

IMAGE_NAME=pi:lingodb-dev

DOCKER_BUILDKIT=1 docker build --ssh default -t ${IMAGE_NAME} .