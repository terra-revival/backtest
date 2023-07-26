ARG DEPS_IMAGE=ustc_invalid_deps_image

FROM python:3.11.4-alpine3.18 AS python.env

WORKDIR '/app/'

RUN apk update
RUN apk add --no-cache \
  bash \
  make \
  automake \
  gcc \
  g++ \
  linux-headers

RUN addgroup --system 'ustc-untrusted'
RUN adduser --ingroup 'ustc-untrusted' --system 'ustc-python'
RUN chown -R 'ustc-python:ustc-untrusted' '/app/'

ENV PATH "${PATH}:/home/ustc-python/.local/bin"

USER 'ustc-python'

# -----------------------------------------------------------------------------

FROM $DEPS_IMAGE AS env-deps

COPY 'requirements_dev.txt' '/app/'

RUN pip install -r 'requirements_dev.txt' --user --no-cache-dir

# -----------------------------------------------------------------------------

FROM $DEPS_IMAGE AS env-build

COPY '.' '/app/'

RUN make dist install
