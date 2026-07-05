#!/usr/bin/env bash
# Build (and optionally push) all vincentycyao/medvision Docker images.
#
# Usage:
#   ./build_and_push.sh --mode load|push [--only tag1,tag2,...] [--dry-run] [--continue-on-error]
#   ./build_and_push.sh --list
#
# Every dockerfile/Dockerfile.<tag> maps 1:1 to the image tag vincentycyao/medvision:<tag>.
# Dockerfile.base is always built first since every other Dockerfile is
# `FROM vincentycyao/medvision:base`.
#
# --mode load  builds into the local Docker image store (docker buildx --load).
#              Nothing touches Docker Hub; useful for validating a Dockerfile.
# --mode push  builds and pushes straight to Docker Hub (docker buildx --push),
#              overwriting the live tag. Requires `docker login` beforehand.
# --list       print the numbered list of available image tags and exit.
#
# If --only is omitted and this script is run in an interactive terminal, you
# will be prompted to pick which images to build from a numbered list.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

IMAGE="vincentycyao/medvision"
PLATFORM="linux/amd64"
BUILDER_NAME="medvision-amd64"

TAG_SUFFIX=""

mode=""
only=""
dry_run=0
continue_on_error=0
list_only=0

usage() {
    echo "Usage: $0 --mode load|push [--only tag1,tag2,...] [--dry-run] [--continue-on-error]" >&2
    echo "       $0 --list" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --mode)
            mode="$2"; shift 2 ;;
        --only)
            only="$2"; shift 2 ;;
        --dry-run)
            dry_run=1; shift ;;
        --continue-on-error)
            continue_on_error=1; shift ;;
        --list)
            list_only=1; shift ;;
        -h|--help)
            usage ;;
        *)
            echo "Unknown argument: $1" >&2; usage ;;
    esac
done

# Discover all tags: base first, then every other Dockerfile.<tag> in filename order.
all_tags=("base")
for f in Dockerfile.*; do
    tag="${f#Dockerfile.}"
    [ "$tag" = "base" ] && continue
    all_tags+=("$tag")
done

if [ "$list_only" -eq 1 ]; then
    echo "Available images (dockerfile/Dockerfile.<tag>):"
    i=1
    for t in "${all_tags[@]}"; do
        printf '  %2d) %s\n' "$i" "$t"
        i=$((i + 1))
    done
    exit 0
fi

if [ "$mode" != "load" ] && [ "$mode" != "push" ]; then
    echo "ERROR: --mode must be 'load' or 'push'" >&2
    usage
fi

# If the caller didn't pin down --only and we're attached to a terminal, let
# them pick from a numbered list instead of building everything.
if [ -z "$only" ] && [ -t 0 ]; then
    echo "Available images (dockerfile/Dockerfile.<tag>):"
    i=1
    for t in "${all_tags[@]}"; do
        printf '  %2d) %s\n' "$i" "$t"
        i=$((i + 1))
    done
    read -r -p "Select images to build (comma-separated numbers or names, blank for all): " selection
    only="$selection"
fi

# Apply --only filter (or interactive selection), preserving base-first ordering.
if [ -n "$only" ]; then
    IFS=',' read -r -a wanted <<< "$only"
    # Trim whitespace from each entry so "1, 2, eval_claude" works.
    for idx in "${!wanted[@]}"; do
        wanted[$idx]="$(echo "${wanted[$idx]}" | xargs)"
    done
    tags=()
    for i in "${!all_tags[@]}"; do
        t="${all_tags[$i]}"
        num="$((i + 1))"
        for w in "${wanted[@]}"; do
            if [ "$t" = "$w" ] || [ "$num" = "$w" ]; then
                tags+=("$t")
                break
            fi
        done
    done
else
    tags=("${all_tags[@]}")
fi

if [ "${#tags[@]}" -eq 0 ]; then
    echo "ERROR: no matching tags to build (check --only/selection against dockerfile/Dockerfile.* filenames)" >&2
    exit 1
fi

if [ "$mode" = "push" ]; then
    # The default Docker-Desktop `docker` driver's support for --push (without
    # --load) isn't guaranteed across configs. A docker-container builder makes
    # --push reliable, and means children always resolve `FROM ...:base` by
    # pulling it from the registry -- no local/registry cache ambiguity.
    if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
        echo "==> Creating docker-container buildx builder '${BUILDER_NAME}' (required for reliable --push)"
        docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use
    else
        docker buildx use "${BUILDER_NAME}"
    fi
fi

mode_flag="--load"
[ "$mode" = "push" ] && mode_flag="--push"

echo "==> Mode: ${mode} | Platform: ${PLATFORM} | Tags: ${tags[*]}"

failed=()
succeeded=()

for tag in "${tags[@]}"; do
    dockerfile="Dockerfile.${tag}"
    if [ ! -f "$dockerfile" ]; then
        echo "ERROR: ${dockerfile} not found, skipping ${tag}" >&2
        failed+=("$tag")
        continue
    fi

    cmd=(docker buildx build --platform "${PLATFORM}" -f "${dockerfile}" -t "${IMAGE}:${tag}${TAG_SUFFIX}" "${mode_flag}" .)

    echo "==> [$tag] ${cmd[*]}"
    if [ "$dry_run" -eq 1 ]; then
        continue
    fi

    if "${cmd[@]}"; then
        succeeded+=("$tag")
    else
        failed+=("$tag")
        if [ "$tag" = "base" ]; then
            echo "ERROR: base image build failed; every other image depends on it, aborting." >&2
            break
        fi
        if [ "$continue_on_error" -eq 0 ]; then
            echo "ERROR: build failed for ${tag}; aborting (pass --continue-on-error to keep going)" >&2
            break
        fi
        echo "WARNING: build failed for ${tag}; continuing (--continue-on-error)" >&2
    fi
done

if [ "$dry_run" -eq 1 ]; then
    echo "==> Dry run complete, nothing was built."
    exit 0
fi

echo "==> Done. Succeeded: ${#succeeded[@]}/${#tags[@]}"
[ "${#succeeded[@]}" -gt 0 ] && printf '    OK   %s\n' "${succeeded[@]}"
[ "${#failed[@]}" -gt 0 ] && printf '    FAIL %s\n' "${failed[@]}"

if [ "${#failed[@]}" -gt 0 ]; then
    exit 1
fi
