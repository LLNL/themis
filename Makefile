SHELL := /bin/bash

USER_WORKSPACE := $(if $(USER_WORKSPACE), $(USER_WORKSPACE),/usr/workspace/$(USER))
WORKSPACE = $(USER_WORKSPACE)/gitlab/weave/themis
THEMIS_ENV := $(if $(THEMIS_ENV), $(THEMIS_ENV),themis_venv)

CZ_GITLAB = "ssh://git@czgitlab.llnl.gov:7999"
RZ_GITLAB = "ssh://git@rzgitlab.llnl.gov:7999"
PROJECT = "weave/themis.git"

# PIP_OPTIONS = --trusted-host www-lc.llnl.gov
PIP_OPTIONS = --trusted-host wci-repo.llnl.gov --index-url https://wci-repo.llnl.gov/repository/pypi-group/simple --use-pep517

setup:
	@[ -d $(WORKSPACE) ] || mkdir -p $(WORKSPACE);


.PHONY: create_env
create_env: setup
	@echo "Create venv for running themis...workspace: $(WORKSPACE)"
	cd $(WORKSPACE); \
	if [ -d $(THEMIS_ENV) ]; then rm -rf $(THEMIS_ENV); fi; \
	/usr/tce/packages/python/python-3.8.2/bin/python3 -m venv $(THEMIS_ENV); \
	source $(THEMIS_ENV)/bin/activate && \
	pip install $(PIP_OPTIONS) --upgrade pip && \
	pip install $(PIP_OPTIONS) --upgrade setuptools && \
	pip install $(PIP_OPTIONS) --force pytest && which pytest
	@echo "Themis virtual env is created: $(WORKSPACE)/$(THEMIS_VENV)"

.PHONY: install
install:
	@echo "Install Themis into venv $(WORKSPACE)/$(THEMIS_ENV)..."
	source $(WORKSPACE)/$(THEMIS_ENV)/bin/activate && \
	pip install $(PIP_OPTIONS) . && \
	pip list

.PHONY: run_unit_tests
run_unit_tests:
	@echo "Run Themis unit tests..."
	source $(WORKSPACE)/$(THEMIS_ENV)/bin/activate && cd tests/unit && \
	if [ "$(UNIT_TESTS)" ]; then \
		for t in $(UNIT_TESTS); do \
			pytest -vv --capture=tee-sys $$t.py; \
		done; \
	else \
		pytest -vv --capture=tee-sys test_*.py; \
	fi

.PHONY: run_integration_tests
run_integration_tests:
	@echo "Run Themis integration tests..."
	source $(WORKSPACE)/$(THEMIS_ENV)/bin/activate && cd tests/integration && \
	if [ "$(INTEGRATION_TESTS)" ]; then \
		for t in $(INTEGRATION_TESTS); do \
			pytest -vv --capture=tee-sys $$t.py; \
		done; \
	else \
		pytest -vv --capture=tee-sys test_*.py; \
	fi






