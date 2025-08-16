FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src

# train | evaluate | both
ENV MODE=train

CMD if [ "$MODE" = "evaluate" ]; then         echo "Avvio valutazione...";         python src/fashion_mnist/evaluate.py;     elif [ "$MODE" = "both" ]; then         echo "Avvio addestramento...";         python src/fashion_mnist/train.py &&         echo "Avvio valutazione...";         python src/fashion_mnist/evaluate.py;     else         echo "Avvio addestramento...";         python src/fashion_mnist/train.py;     fi
