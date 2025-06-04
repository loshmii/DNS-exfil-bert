## Conda Environment Activation

This repository uses a preinstalled conda environment located at `.conda_env`. Before executing any code or running scripts, the agent must ensure that the correct environment is activated and that all Python dependencies are available.

### **How to Activate the Conda Environment**

1. **Initialize Conda in the Shell**

   Before you can use `conda activate`, you must load the Conda shell functions. In a new shell, do:

   ```bash
   source ./miniconda3/etc/profile.d/conda.sh
   ```

   > *Note: This step is required for `conda` commands to work, as they are not available in the system path by default after the preinstall script.*

2. **Activate the Environment by Path**

   Activate the conda environment using its absolute or relative path:

   ```bash
   conda activate $(pwd)/.conda_env
   ```

   Or, if you are already in the repository root directory:

   ```bash
   conda activate ./conda_env
   ```

   or

   ```bash
   conda activate .conda_env
   ```

   > *Tip: Do not use only the environment name. Always use the path.*

3. **Verify Environment Activation**

   After activation, verify that the correct Python and packages are available:

   ```bash
   which python
   python -c "import sys; print(sys.executable)"
   python -c "import torch, transformers"
   ```

   These should run without errors and show paths inside `.conda_env`.

### **Alternative: Run Python Directly from Environment**

If you cannot use `conda activate`, you can run Python from the environment directly:

```bash
./.conda_env/bin/python
```

All installed packages will be available when using this Python executable.

---

## **Summary Checklist**

* [ ] `source ./miniconda3/etc/profile.d/conda.sh`
* [ ] `conda activate $(pwd)/.conda_env`
* [ ] Run code as needed (or use `./.conda_env/bin/python` directly)
