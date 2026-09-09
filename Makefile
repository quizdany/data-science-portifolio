.PHONY: install train predict test lint clean

install:
	python -m pip install -e ".[dev]"

train:
	python -m penguin_classifier train

predict:
	python -m penguin_classifier predict --sample-json "{\"bill_length_mm\":39.1,\"bill_depth_mm\":18.7,\"flipper_length_mm\":181,\"body_mass_g\":3750,\"island\":\"Torgersen\",\"sex\":\"Male\"}"

test:
	pytest

lint:
	ruff check src tests

clean:
	python -m penguin_classifier clean

