Project: MLP Alzheimer MRI Classifier

What this is
- A neural network to tell apart:
  - AD: Alzheimer’s Disease
  - CN: Cognitive Normal

What is included
- train_mlp_adni.py   
- README.txt          
- Makefile            
- report.pdf          


Requirements
- Python 3
- torch
- torchvision
- nibabel
- numpy
- scikit learn

Example install:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install nibabel numpy scikit-learn

How to run

From project_root, after you put the ADNI folder there:

Option 1:
make run

Option 2:
python3 train_mlp_adni.py --data_root .
