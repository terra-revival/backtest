#!/bin/bash

set -f -o pipefail

source_dir="$( dirname -- "$( readlink -f -- "${0}"; )"; )"

# Usage:
# ops::shasums::gen_from_file 'README.md'
# ops::shasums::gen_from_file './README.md'
# ops::shasums::gen_from_file '/absolute/path/README.md'

ops::shasums::gen_from_file () {
  local -r shasum_file="$(echo ${1} | xargs)"

  if [[ ! "${shasum_file}" ]]; then
    return 1
  fi

  if [[ ! -f "${shasum_file}" ]]; then
    return 2
  fi

  local -r shasum="$(
    cat "${shasum_file}" | \
      openssl dgst -sha256 | \
      sed 's|^.* ||'
  )"

  echo "${shasum:0:8}"
}

ops::shasums::gen_from_current_git_commit_hash () {
  local -r git_commit_id="$(git -C "${source_dir}" rev-parse HEAD)"
  echo "${git_commit_id:0:8}"
}

ops::shasums::refresh_docker_compose_env_tags () {
  local -r env_file="${source_dir}/ops/Dockerfile"
  local -r env_hash="$(ops::shasums::gen_from_file "${env_file}")"

  local -r deps_file="${source_dir}/requirements_dev.txt"
  local -r deps_hash="$(ops::shasums::gen_from_file "${deps_file}")"

  local -r build_hash="$(ops::shasums::gen_from_current_git_commit_hash)"

  local -r compose_env_file="${source_dir}/.env"

  echo "source_dir: ${source_dir}"

  echo "env_hash: ${env_hash}"
  echo "deps_hash: ${deps_hash}"
  echo "build_hash: ${build_hash}"

  echo "compose_env_file: ${compose_env_file}"

  if [[ ! -f "${compose_env_file}" ]]; then
    cat > "${compose_env_file}" << EOF
ENV_TAG=${env_hash}
DEPS_TAG=${deps_hash}
BUILD_TAG=${build_hash}
EOF
    return
  fi

  sed -i-backup \
    -e "s|ENV_TAG=.*|ENV_TAG=${env_hash}|g" \
    -e "s|DEPS_TAG=.*|DEPS_TAG=${deps_hash}|g" \
    -e "s|BUILD_TAG=.*|BUILD_TAG=${build_hash}|g" \
    "${compose_env_file}"
}
