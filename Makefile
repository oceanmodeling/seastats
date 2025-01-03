.PHONY: list docs

list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | grep -E -v -e '^[^[:alnum:]]' -e '^$@$$'

init:
	poetry install --sync
	pre-commit install

style:
	pre-commit run ruff-format -a

lint:
	pre-commit run ruff -a

mypy:
	dmypy run seastats

deps:
	pre-commit run poetry-lock -a
	pre-commit run poetry-export -a

cov:
	coverage erase
	python -m pytest \
		-vv \
		--durations=10 \
		--cov=seastats \
		--cov-report term-missing
