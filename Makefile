.PHONY: all binary build default shell test help

# get OS/Arch of docker engine
DOCKER_OSARCH := $(shell bash -c 'source build/detect-daemon-osarch && echo $${DOCKER_ENGINE_OSARCH:-$$DOCKER_CLIENT_OSARCH}')
DOCKERFILE := $(shell bash -c 'source build/detect-daemon-osarch && echo $${DOCKERFILE}')

CUDA_VERSION := $(if $(CUDA_VERSION),$(CUDA_VERSION),10.0)
DOCKER_BUILD_ARGS += --build-arg CUDA_VERSION="$(CUDA_VERSION)"

# env vars passed through directly to Docker's build scripts
DOCKER_ENVS := \
    -e SKIP_TESTS \
    -e CUDA_VERSION="$(CUDA_VERSION)" \
    -e CMAKE_OPTS="$(CMAKE_OPTS)"

# to allow `make BIND_DIR=. shell` or `make BIND_DIR= test`
# (default to no bind mount if DOCKER_HOST is set)
BIND_DIR := $(if $(BINDDIR),$(BINDDIR),bundles)
DOCKER_MOUNT := $(if $(BIND_DIR),-v "$(CURDIR)/$(BIND_DIR):/devel/dlf/$(BIND_DIR)")

GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null)
GIT_BRANCH_CLEAN := $(shell echo $(GIT_BRANCH) | sed -e "s/[^[:alnum:]]/-/g")
DOCKER_IMAGE := dlf-dev$(if $(GIT_BRANCH_CLEAN),:$(GIT_BRANCH_CLEAN))
DOCKER_FLAGS := docker run --rm -i --privileged $(DOCKER_ENVS) $(DOCKER_MOUNT)

# if this session isn't interactive, then we don't want to allocate a
# TTY, which would fail, but if it is interactive, we do want to attach
# so that the user can send e.g. ^C through.
INTERACTIVE := $(shell [ -t 0 ] && echo 1 || echo 0)
ifeq ($(INTERACTIVE),1)
    DOCKER_FLAGS += -t
endif

DOCKER_RUN_DOCKER := $(DOCKER_FLAGS) "$(DOCKER_IMAGE)"

default: build
	$(DOCKER_RUN_DOCKER) build/make.sh

all: build ## build linux binaries, run all test
	$(DOCKER_RUN_DOCKER) build/make.sh

build: bundles
	docker build ${DOCKER_BUILD_ARGS} -t "$(DOCKER_IMAGE)" -f "$(DOCKERFILE)" .

bundles:
	mkdir bundles

shell: build ## start a shell inside the build env
	$(DOCKER_RUN_DOCKER) bash

clean:
	@docker images | grep '<none>' | awk '{print $3}' | xargs docker rm 2>/dev/null || true
	@rm -rf bundles/*

help: ## this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {sub("\\\\n",sprintf("\n%21c"," "), $$@);printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
