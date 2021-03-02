#!/bin/sh

# Option 1: Cross-platform build
docker buildx build --platform linux/arm --tag sixsq/tensorflow-lite-object-detector:latest-arm -f Dockerfile.arm .
docker buildx build --platform linux/amd64 --tag sixsq/tensorflow-lite-object-detector:latest-amd64 -f Dockerfile.amd64 .

# Option 2: Build on target platform
## arm
docker build -f Dockerfile.arm -t sixsq/tensorflow-lite-object-detector:latest-arm .
docker push sixsq/tensorflow-lite-object-detector:latest-arm
## amd64
docker build -f Dockerfile.amd64 -t sixsq/tensorflow-lite-object-detector:latest-amd64 .
docker push sixsq/tensorflow-lite-object-detector:latest-arm

# Create multiplatform manifest
docker manifest create sixsq/tensorflow-lite-object-detector:latest sixsq/tensorflow-lite-object-detector:latest-amd64 sixsq/tensorflow-lite-object-detector:latest-arm
docker manifest push sixsq/tensorflow-lite-object-detector:latest
