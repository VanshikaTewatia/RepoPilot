# Bakes pytest (and setuptools/wheel, the PEP 517 build backend pip needs to
# fetch even for a zero-dependency project) into the Python sandbox image, so
# the isolated, network_mode="none" verification container never needs
# network access just to run the test harness itself.
#
# python:3.11-slim ships no test framework. Installing pytest at
# verification-run time would need network the sandbox correctly denies --
# and even a dependency-free project's `pip install .` needs to fetch a PEP
# 517 build backend (setuptools/wheel) unless one is already present, which
# also fails offline. Both are pre-installed here instead, at build time
# (real network access, same as any `docker pull`), so nothing at
# verification-run time needs the network for the harness itself.
#
# A target repository's OWN third-party dependencies are handled separately
# (see VerificationEngine._install_python_deps_for_docker: installed
# host-side, outside this image/container entirely, then mounted read-only)
# -- never baked into this image, which stays generic across every Python
# repository RepoPilot verifies.
#
# Build once, ahead of time:
#   docker build -t repopilot-sandbox-python:3.11 docker/sandbox/python
FROM python:3.11-slim

RUN pip install --no-cache-dir --disable-pip-version-check pytest setuptools wheel
