#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = Decision-Trees-Implementation
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python
ENV_NAME = CART-Decision-Tree
PYTHON = conda run -n $(ENV_NAME) python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	conda env update --name $(ENV_NAME) --file environment.yml --prune

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 CART_Decision_Tree
	isort --check --diff --profile black CART_Decision_Tree
	black --check --config pyproject.toml CART_Decision_Tree

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml CART_Decision_Tree

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(ENV_NAME) -f environment.yml

	@echo ">>> conda env created. Activate with:\nconda activate $(ENV_NAME)"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# Train the model
.PHONY: train
train:
	@echo "Training the model..."
	$(PYTHON) CART_Decision_Tree/modeling/train.py

# Predict using the model
.PHONY: predict
predict:
	@echo "Predicting..."
	$(PYTHON) CART_Decision_Tree/modeling/predict.py


.PHONY: search
search:
	@echo "searching..."
	$(PYTHON) CART_Decision_Tree/modeling/GridSearch.py


.PHONY: plot
plot:
	@echo "plotting..."
	$(PYTHON) CART_Decision_Tree/plots.py
#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
