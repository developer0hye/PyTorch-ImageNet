def count_total_prameters(model):
 return sum(p.numel() for p in model.parameters())