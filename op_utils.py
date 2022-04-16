def pytorch_count_params(model, trainable=True):
    "count number trainable parameters in a pytorch model"
    s = 0
    for p in model.parameters():
        if trainable:
            if not p.requires_grad: continue
        try:
            s += p.numel()
        except:
            pass
    return s