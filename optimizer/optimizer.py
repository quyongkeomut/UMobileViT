from torch.optim import Adam, AdamW, NAdam, SGD

OPTIMIZERS = {
    "adam": Adam,
    "adamw": AdamW,
    "nadam": NAdam,
    "sgd": SGD
}

