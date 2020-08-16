# How to run VAMPIRE with SCHOLAR-esque covariates

## Step 1: Gather data

We have a toy dataset in `examples/toy` that will help us test adding covariates. Similarly, 
add in a dataset that contains a train.jsonl and a dev.jsonl to tokenize the data more easily. 

Only difference for tapt-selection is that it's called `data`, not `examples`

## Step 2: Pretokenize data

```bash
python -m scripts.pretokenizer --input-file examples/toy/train.jsonl --tokenizer spacy --json --lower --num-workers 8 --worker-tqdms 8 --output-file examples/toy/train.tok.jsonl
python -m scripts.pretokenizer --input-file examples/toy/dev.jsonl --tokenizer spacy --json --lower --num-workers 8 --worker-tqdms 8 --output-file examples/toy/dev.tok.jsonl
```

## Step 3: Preprocess data

```bash
python -m scripts.preprocess_data \
            --train-path examples/toy/train.jsonl \
            --dev-path examples/toy/dev.jsonl \
            --tokenize \
            --tokenizer-type spacy \
            --vocab-size 50000 \
            --serialization-dir examples/toy \
            --preprocess-covariates True
```

## Step 4: Train VAMPIRE

```bash
export DATA_DIR="$(pwd)/examples/toy"
export VOCAB_SIZE={vocab_size}
```

```bash
python -m scripts.train \
            --config training_config/vampire.jsonnet \
            --serialization-dir model_logs/vampire \
            --environment VAMPIRE \
            --device -1 --seed 42
```