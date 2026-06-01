.PHONY: install test schema-audit smoke full figures results closeout secret-audit

install:
	python -m pip install -e ".[dev,ml]"

test:
	python -m pytest

schema-audit:
	python -m ctrsdf.pipeline schema-audit --config configs/project.yaml

smoke:
	python -m ctrsdf.pipeline smoke --config configs/project.yaml

full:
	python -m ctrsdf.pipeline full --config configs/project.yaml

figures:
	python -m ctrsdf.pipeline figures --config configs/project.yaml

results:
	python -m ctrsdf.pipeline results --config configs/project.yaml

closeout:
	python -m ctrsdf.pipeline results --config configs/project.yaml

secret-audit:
	python -m ctrsdf.utils.secret_audit --root .
