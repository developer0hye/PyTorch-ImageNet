# PyTorch-ImageNet


## Preprocessing

### Normalization
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. You can use the following transform to normalize:

```
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
```

## Pretrained models

| Model    |#Params| Weight File Size(Byte)|  Top1Acc(%)|
|--------------|-----------|-----------|-----------|
|[YOLOv3TinyBackbone(=tiny)]() | 7,323,487 |29,309,647|58.97