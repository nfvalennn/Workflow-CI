FROM continuumio/miniconda3

WORKDIR /app

COPY conda.yaml .
RUN conda env create -f conda.yaml

COPY . /app

CMD ["mlflow", "run", "."]
