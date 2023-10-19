Generate validation features:

```
TORCHAUDIO_USE_FFMPEG=0 TORCHAUDIO_USE_SOX=0 PYTORCH_ENABLE_MPS_FALLBACK=1 python dump_hubert_feature.py ../../../../metadata valid ../../../../hubert_base_ls960.pt 5 12 0 ../../../../features
```

# K-means clustering

```
python learn_kmeans.py ../../../../features valid 12 ../../../../valid.km 100 --percent -1
```
