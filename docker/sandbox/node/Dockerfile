# Optional, richer Node verification image: everything NodeAdapter's default
# "node:20-slim" already gives you (npm), plus pnpm and yarn pre-activated
# via corepack so lockfile-selected pnpm/yarn projects can be verified fully
# offline (the sandbox container runs with network_mode="none" by default).
#
# NodeAdapter defaults to the stock "node:20-slim" image on purpose: it's
# always pullable with no build step and covers the common npm case (this is
# what fixes "npm: not found" for a plain Node/React repo). This image is an
# opt-in upgrade for repositories whose lockfile selects pnpm or yarn --
# point NodeAdapter.docker_image at the tag you build this under.
#
# Build once, ahead of time (needs network access, same as any `docker pull`):
#   docker build -t repopilot-sandbox-node:20-full docker/sandbox/node
#
# corepack prepare --activate downloads and caches the actual pnpm/yarn
# binaries into this image layer at build time, as root, with build-time
# network access -- so no network or write-permission workaround is needed
# at verification-run time, when the container runs as an unprivileged,
# network-isolated user.
FROM node:20-slim

RUN corepack enable \
    && corepack prepare pnpm@latest --activate \
    && corepack prepare yarn@latest --activate
