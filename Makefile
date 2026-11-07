# Common tasks. The LLM experiments need torch/transformers/peft (requirements.txt).
.PHONY: test solve tabular figures

test:
	python3 -m pytest -q

solve:
	python3 scripts/solve.py leduc

tabular:
	PYTHONPATH=. python3 scripts/train_tabular.py --seeds 3

figures:
	PYTHONPATH=. python3 scripts/plots.py
