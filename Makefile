PYTHON = python3
DATA_ROOT = .

run:
	$(PYTHON) train_mlp_adni.py --data_root $(DATA_ROOT)
