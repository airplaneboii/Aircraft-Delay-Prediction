# Aircraft Delay Prediction
Group project for MLG course 2025.

This is still work in progress.

---

## Install all required packages using:
```bash
pip install -r requirements.txt
```

---
## Run the code with
```bash
python main.py
```
To adjust model parameters, as well to select to train or test model, use additional `argparse` arguments. All of them are listed in `src/config.py`. Example of how to run code using those arguments:
```bash
python main.py --mode train --model_type rgcnmodel --epochs 100 --lr 0.0005
```

---
## Editing
To add model, add new `.py` code of your model to `src/models/` folder. To add graph, add new `.py` code of your model to `src/graph/` folder. Following python structure of other models/graphs is strongly encouraged. Additionally, filename should be added to argument list in `src/config.py`. Simmilarly, `main.py` should be updated (new `elif` option should be properly added).

Before submitting new model or graph, make sure code runs without errors, failures and warnings.