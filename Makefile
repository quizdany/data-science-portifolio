.PHONY: install train predict credit-train credit-score test lint clean

install:
	python -m pip install -e ".[dev]"

train:
	python -m penguin_classifier train

predict:
	python -m penguin_classifier predict --sample-json "{\"bill_length_mm\":39.1,\"bill_depth_mm\":18.7,\"flipper_length_mm\":181,\"body_mass_g\":3750,\"island\":\"Torgersen\",\"sex\":\"Male\"}"

credit-train:
	python -m credit_risk train

credit-score:
	python -m credit_risk score --applicant-json "{\"duration\":24,\"amount\":4000,\"installment_rate\":3,\"present_residence\":2,\"number_credits\":1,\"people_liable\":1,\"status\":\"... < 100 DM\",\"credit_history\":\"existing credits paid back duly till now\",\"purpose\":\"car (new)\",\"savings\":\"... < 100 DM\",\"employment_duration\":\"1 <= ... < 4 years\",\"other_debtors\":\"none\",\"property\":\"car or other\",\"other_installment_plans\":\"none\",\"housing\":\"rent\",\"job\":\"skilled employee/official\",\"telephone\":\"no\"}"

test:
	pytest

lint:
	ruff check src tests

clean:
	python -m penguin_classifier clean

