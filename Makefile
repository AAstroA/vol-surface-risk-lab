.PHONY: install demo probe download run wrds long test clean

install:
	python3 -m pip install -e ".[dev]"

demo:
	python3 -m spx_risk demo --config configs/demo.yaml

probe:
	python3 -m spx_risk probe-wrds --config configs/default.yaml

download:
	python3 -m spx_risk download --config configs/default.yaml

run:
	python3 -m spx_risk run --config configs/default.yaml

wrds:
	python3 -m spx_risk all --config configs/default.yaml

long:
	python3 -m spx_risk all --config configs/long_horizon.yaml

test:
	python3 -m pytest

clean:
	python3 -m spx_risk clean-generated --config configs/default.yaml
