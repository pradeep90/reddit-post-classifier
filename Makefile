VENV := venv/bin
PYTHON := $(VENV)/python
UNAME := $(shell uname -n)
SHELL := /bin/bash

test: src/utils.py src/pretrained_word_embeddings.py
	cd src && ../$(PYTHON) -m unittest utils.UtilsTest pretrained_word_embeddings.CNNTest

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

.PHONY: train_and_test_model
train_and_test_model: src/train_and_test_model.py
	(echo; echo -n 'Start: '; date; $(PYTHON) $^; echo -n 'End: '; date; echo) | tee -a logs/model_outputs.txt

.PHONY: pretrained_word_embeddings
pretrained_word_embeddings: src/pretrained_word_embeddings.py
	(echo; echo -n 'Start: '; date; $(PYTHON) $^; echo -n 'End: '; date; echo) | tee >(grep -v 'ETA' >> logs/model_outputs.txt)

.PHONY: test_preprocessing
test_preprocessing: src/preprocessing/tests/test_preprocessing.py
	$(PYTHON) -m unittest -v $^

.PHONY: all
all: test report
