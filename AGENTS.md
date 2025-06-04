## Conda Environment Setup and Activation

This repository requires a specific Conda environment to ensure all dependencies are met for reproducible execution. The environment is defined in `environment.yml` and should be created at the repository root as `.conda_env`.
**The following steps describe how to prepare and use the environment—both for first-time setup and daily usage.**

---

### **Step 1: Download and Install Miniconda**

If you are running inside a container (or on a new system), you will need to install Miniconda first:

```bash
# Set your desired install directory (example: /workspace/DNS-exfil-bert)
export EXPDIR=$(pwd)
export CONDA_DIR=$EXPDIR/miniconda3

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR

# Load Conda functions into the shell
source $CONDA_DIR/etc/profile.d/conda.sh
```

> **Note:** `conda` commands are not available in your PATH by default after install. Always use `source $CONDA_DIR/etc/profile.d/conda.sh` before calling `conda`.

---

### **Step 2: Create the Environment from `environment.yml`**

**Important:** The environment should NOT be inside the Miniconda directory.
Instead, create it as a *directory-based* environment at the repository root.

```bash
# Set environment prefix (path to where env will live)
ENV_PREFIX="${EXPDIR}/.conda_env"

# Create environment from YAML (if not already created)
conda env create -p "${ENV_PREFIX}" -f "${EXPDIR}/environment.yml"
```

---

### **Step 3: Activate the Conda Environment**

You must *initialize* Conda in every new shell **before** activating:

```bash
source ./miniconda3/etc/profile.d/conda.sh
conda activate $(pwd)/.conda_env
```

**Tips:**

* Always use the *path* (`.conda_env` or `$(pwd)/.conda_env`), not just the name.
* You can check the activation:

  ```bash
  which python
  python -c "import sys; print(sys.executable)"
  python -c "import torch, transformers"
  ```

  All should point to locations under `.conda_env`.

---

### **Alternative: Run Python Directly (No Activation)**

If you cannot use `conda activate` (e.g., in some CI/CD, Docker, or scripts),
run the Python interpreter from the environment directly:

```bash
./.conda_env/bin/python script.py
```

All dependencies from `environment.yml` will be available.

---

## **Summary Checklist for Agents/Users**

* [ ] Download and install Miniconda if not present.
* [ ] `source ./miniconda3/etc/profile.d/conda.sh`
* [ ] Create the environment:
  `conda env create -p $(pwd)/.conda_env -f environment.yml`
* [ ] Activate:
  `conda activate $(pwd)/.conda_env`
* [ ] (Or run via `./.conda_env/bin/python ...`)

---

## **Automated Setup (Recommended Script)**

For convenience, the following script will fully automate setup and activation.
Place in your project root and run:

```bash
#!/usr/bin/env bash
set -euo pipefail

EXPDIR=$(pwd)
CONDA_DIR=$EXPDIR/miniconda3
ENV_PREFIX="${EXPDIR}/.conda_env"

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR

source $CONDA_DIR/etc/profile.d/conda.sh
conda env create -p "${ENV_PREFIX}" -f "${EXPDIR}/environment.yml"
conda activate "${ENV_PREFIX}"

echo "✅ Conda environment installed and activated at ${ENV_PREFIX}"
```

---

### **Note for Agents**

* If you want to use any Python libraries required by the project, you must follow the above steps **inside the container** or host machine.
* All installed dependencies will be available after environment creation and activation.
* The environment can be used both interactively and in automated scripts.