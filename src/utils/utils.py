

def get_perplexity(loss, base=2):
    if loss is None:
        return 0.
    return base**loss
