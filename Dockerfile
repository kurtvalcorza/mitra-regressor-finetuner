# DIMER finetuner — Mitra tabular regression (GPU).
# AutoGluon's Mitra runs on torch/CUDA; use a CUDA runtime base so the 5070 Ti / cluster
# GPU is usable. Match the CUDA minor to the cluster's driver.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System libs AutoGluon's stack occasionally needs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# OPTION A (recommended for reproducibility): bake pinned Mitra weights into the image so
# runs never depend on a network fetch and cannot drift between builds.
#   ENV HF_HOME=/opt/hf
#   RUN python -c "from huggingface_hub import snapshot_download; \
#       snapshot_download('autogluon/mitra-regressor', \
#       revision='5f277aa8f69042d39d6ac3612aed18bb9279bd95')"
# OPTION B: let AutoGluon download Mitra at runtime (simpler, but needs network egress and
# is unpinned unless you also set the revision). Leave as-is to use B.

# Bake our own task type so the Custom/Other -> object_detection normalization can't win.
ENV DIMER_TASK_TYPE=tabular_regression

COPY train.py ./

CMD ["python", "train.py"]
