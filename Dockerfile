FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# canale ufficiale CPU di PyTorch
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src

# train | evaluate | both
ENV MODE=train

CMD if [ "$MODE" = "evaluate" ]; then         echo "Avvio valutazione...";         python src/evaluate.py;     elif [ "$MODE" = "both" ]; then         echo "Avvio addestramento...";         python src/train.py &&         echo "Avvio valutazione...";         python src/evaluate.py;     else         echo "Avvio addestramento...";         python src/train.py;     fi
