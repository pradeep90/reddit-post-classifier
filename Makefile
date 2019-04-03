VENV := venv/bin
PYTHON := $(VENV)/python
UNAME := $(shell uname -n)

.PHONY: set_up_venv
set_up_venv:
	python3 -m venv venv

.PHONY: install_packages
install_packages:
	$(VENV)/pip install -r requirements.txt

.PHONY: freeze
freeze:
	$(VENV)/pip freeze > requirements.txt

.PHONY: explore
explore: src/explore_data.py
	$(PYTHON) $^

.PHONY: kaggle_nbc
kaggle_nbc: src/kaggle_nbc.py
	$(PYTHON) $^

.PHONY: all
all: test report
