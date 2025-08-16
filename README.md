# Fashion-MNIST CNN — Docker + CI

La repository contiene :
- CNN semplice per Fashion-MNIST
- Esecuzione CPU-only
- Dockerfile con MODE=train|evaluate|both
- CI GitHub Actions (lint, test, build, training + artifact)

## Setup locale (opzionale)
```bash
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

## Docker
Build:
```bash
docker build -t fashion-mnist .
```

Solo training:
```bash
docker run --rm -e MODE=train fashion-mnist
```

Solo evaluation (richiede `src/model.pth` creato da un training precedente nello stesso container):
```bash
docker run --rm -e MODE=evaluate fashion-mnist
```

Training + evaluation nello stesso container (consigliato):
```bash
docker run --rm -e MODE=both fashion-mnist
```

## CI (GitHub Actions)
- Push su `main` → lint, test, build Docker, training nel container.
- L’artifact `trained-model` contiene `model.pth`.

