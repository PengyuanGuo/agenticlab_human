# Installation Guide of Anygrasp (Open Sourced Version)

Below is a clean step-by-step installation plan for **GraspNet / AnyGrasp-style inference with MinkowskiEngine** on a **new PC with RTX 3080**.

For RTX 3080, use:

```bash
export TORCH_CUDA_ARCH_LIST="8.6"
```

---

# GraspNet / AnyGrasp Installation Plan

## 0. Create environment

```bash
conda create -n anygrasp python=3.10 -y
conda activate anygrasp
python -m pip install --upgrade pip
```

---

## 1. Install PyTorch with CUDA 12.1

```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Expected:

```
Torch: 2.4.0+cu121
Torch CUDA: 12.1
CUDA available: True
GPU: RTX 3080
```

---

## 2. Install basic build dependencies

```bash
pip install ninja cmake wheel "numpy==1.26.*"
```

Then force compatible setuptools and numpy:

```bash
pip install --force-reinstall "setuptools<60" "numpy==1.26.4" wheel ninja cmake
```

Verify:

```bash
python -V
python -c "import setuptools, numpy; print('setuptools=', setuptools.__version__, 'numpy=', numpy.__version__)"
```

---

## 3. Install compiler and CUDA toolchain

Use **conda CUDA nvcc**, not a mixed pip/conda CUDA toolchain.

```bash
conda install -c conda-forge gcc_linux-64=11.* gxx_linux-64=11.* cuda-nvcc=12.1 -y
conda install -c anaconda openblas-devel -y
```

Install the full CUDA dev set:

```bash
conda install -c conda-forge \
"cuda-version=12.1" \
cuda-nvcc \
cuda-cudart-dev \
cuda-nvrtc-dev \
libcublas-dev \
libcusparse-dev \
libcurand-dev \
libcusolver-dev \
cuda-nvtx-dev \
-y
```

---

## 4. Set build environment variables

Check first:

```bash
echo "CC=$CC"
echo "CXX=$CXX"
which nvcc
nvcc --version
```

Then set:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=8
export BLAS_INCLUDE_DIRS="$CONDA_PREFIX/include"
export BLAS_LIBRARY_DIRS="$CONDA_PREFIX/lib"
export LDFLAGS="-Wl,-rpath,$BLAS_LIBRARY_DIRS $LDFLAGS"
```

For RTX 3080:

```bash
export TORCH_CUDA_ARCH_LIST="8.6"
```

If `gcc` is not the conda GCC 11 compiler, set CC/CXX explicitly:

```bash
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
```

Verify:

```bash
python - <<'PY'
import torch, shutil
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("nvcc at:", shutil.which("nvcc"))
PY
```

---

## 5. Build and install MinkowskiEngine

```bash
mkdir -p dependencies
cd dependencies

git clone https://github.com/chenxi-wang/MinkowskiEngine.git
cd MinkowskiEngine
git checkout cuda-12-1
```

Clean old builds (if you have any):

```bash
pip uninstall -y MinkowskiEngine || true
python setup.py clean || true
rm -rf build dist *.egg-info MinkowskiEngine.egg-info
```

Install:

```bash
python setup.py install \
--blas_include_dirs=${CONDA_PREFIX}/include \
--blas_library_dirs=${CONDA_PREFIX}/lib \
--blas=openblas
```

Verify:

```bash
python - <<'PY'
import torch
import MinkowskiEngine as ME
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("ME:", ME.__version__ if hasattr(ME, "__version__") else "imported")
print("cuda available:", torch.cuda.is_available())
PY
```

If this passes, the hardest part is done.

---

## 6. Install GraspNet / AnyGrasp dependencies

```bash
pip uninstall -y graspnetAPI sklearn || true
```

Install Python dependencies:

```bash
pip install "numpy==1.26.4" \
scipy \
"transforms3d>=0.4.2" \
open3d \
trimesh \
tqdm \
Pillow \
opencv-python \
matplotlib \
pywavefront \
scikit-image \
autolab_core \
autolab-perception \
cvxopt \
dill \
h5py \
scikit-learn
```

---

## 7. Install GraspNetAPI

If you have the local `graspnetAPI` source:

```bash
cd /path/to/graspnetAPI
pip install --no-deps .
```

If you do not have local source, try:

```bash
pip install graspnetAPI
```

7_2 Install knn and pointnet2

```
cd Path_to_vlm_robobench/vlm_robobench/third_party/graspness_unofficial/knn
python setup.py install

cd Path_to_vlm_robobench/vlm_robobench/third_party/graspness_unofficial/pointnet2
python setup.py install
```

Verify:

```bash
python - <<'PY'
from graspnetAPI import GraspGroup
print("graspnetAPI ok")
PY
```

---

## 8. Install grasp-nms

```bash
python -m pip install grasp-nms
python -c "import grasp_nms; print('grasp_nms ok')"
```

---

## 9. Install project in editable mode

If using your `vlm_robobench` project:

```bash
cd /home/agenticlab/Project/vlm_robobench
pip install -e .
```

Verify import path:

```bash
python - <<'PY'
import vlm_robobench
print(vlm_robobench.__file__)
PY
```

---

## 10. Full smoke test

Run:

```bash
python - <<'PY'
import numpy as np
import torch
import MinkowskiEngine as ME
import open3d as o3d
import cv2
from graspnetAPI import GraspGroup
import grasp_nms

print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("ME ok")
print("open3d:", o3d.__version__)
print("cv2:", cv2.__version__)
print("graspnetAPI ok")
print("grasp_nms ok")
PY
```

Expected:

```
numpy: 1.26.4
torch: 2.4.0+cu121
cuda available: True
ME ok
open3d: 0.19.x
graspnetAPI ok
grasp_nms ok
```

---

## 11. Common fixes

### RTX 3080 CUDA arch

Use:

```bash
export TORCH_CUDA_ARCH_LIST="8.6"
```

For RTX 4090, use:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"
```

### OMP warning from MinkowskiEngine

Optional:

```bash
export OMP_NUM_THREADS=12
```

---

## 12. Recommended final export

After everything works:

```bash
conda env export --no-builds > env_anygrasp_3080_working.yml
pip freeze > pip_anygrasp_3080_working.txt
```

This gives you a reproducible installation record.