export DOCKER_BUILDKIT=1

# Load environment variables (for app)
export ENV_FILE?=.app-env
include ${ENV_FILE}

# For Docker image
export VERSION?=0.1.0
export USERNAME?=$(shell whoami)
export USER_ID?=$(shell id -u)
export GROUP_ID?=$(shell id -g)

export APP_DIR?=app
export INFRA_DIR?=infrastructure
export UML_DIR?=uml-diagram
export DOCKERFILE?=deployment/Dockerfile

# Static code analysis / test
export TEST_DIR?=tests
export TEST_REPORT_DIR?=test_report

build:
	@echo "Building Docker image (MAIN)..."
	docker build \
		--build-arg VERSION=${VERSION} \
		--build-arg USERNAME=${USERNAME} \
		--build-arg USER_ID=${USER_ID} \
		--build-arg GROUP_ID=${GROUP_ID} \
		-t ${USERNAME}/${APP_DIR}:${VERSION} \
		-f ${DOCKERFILE} \
		--no-cache \
		--target main .
	@echo "Building Docker image (TEST)..."
	docker build \
		--build-arg VERSION=${VERSION} \
		--build-arg USERNAME=${USERNAME} \
		--build-arg USER_ID=${USER_ID} \
		--build-arg GROUP_ID=${GROUP_ID} \
		-t ${USERNAME}/${APP_DIR}-test:${VERSION} \
		-f ${DOCKERFILE} \
		--target test .

run:
	@echo "Running Docker container..."
	docker run --rm -it -d --name ${USERNAME}-${APP_DIR} \
		--env-file ${ENV_FILE} \
		${USERNAME}/${APP_DIR}:${VERSION}

clean:
	@echo "Removing Docker container..."
	docker stop ${USERNAME}-${APP_DIR}

# Install Python dependencies in poetry environment
install:
	@echo "Installing Python dependencies..."
	poetry install
	poetry run pre-commit install -f

# Upate Python dependencies in poetry environment
update:
	@echo "Updating Python dependencies..."
	poetry update

test:
	@echo "Running tests..."
	mkdir -p ${TEST_REPORT_DIR}
	poetry run pytest \
		--disable-warnings \
		--cov-report=html:${TEST_REPORT_DIR} \
		--cov=${APP_DIR} .

# Build/Pull infrastructure services
infra-build:
	@echo "Building/Pulling images of infrastructure services..."
	docker-compose -f ${INFRA_DIR}/docker-compose.yml build

# Start infrastructure services
infra-up:
	@echo "Starting infrastructure services..."
	docker-compose -f ${INFRA_DIR}/docker-compose.yml up -d

# Stop infrastructure services
infra-down:
	@echo "Stopping infrastructure services..."
	docker-compose -f ${INFRA_DIR}/docker-compose.yml down

# Show environment variables
app-info:
	@echo "Showing environment variables (used in the project)..."
	@while read -r line; do \
		echo $$line; \
	done < ${ENV_FILE}


help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build: Build Docker image"
	@echo "  run: Run Docker container"
	@echo "  clean: Remove Docker container"
	@echo "  install: Install Python dependencies"
	@echo "  update: Update Python dependencies"
	@echo "  format: Format source code"
	@echo "  test: Run tests"
	@echo "  static-code-analysis: Run static code analysis"
	@echo "  test-report: Generate test report"
	@echo "  infra-build: Build/Pull infrastructure services"
	@echo "  infra-up: Start infrastructure services"
	@echo "  infra-down: Stop infrastructure services"
	@echo "  uml-diagram: Generate UML diagram"
	@echo "  show-env: Show environment variables"
	@echo "  help: Show this message"
