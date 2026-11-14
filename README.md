# Audio-Based Voice Command Recognition System

An end-to-end audio classification project that recognises short spoken commands and supports unknown-word and background-noise detection. The system is designed as a voice-controlled interface for an educational application.

## Project status

**Milestones 1-2: Dataset exploration and preprocessing implementation.** The repository now includes `src/preprocessing.py` and `notebooks/02_preprocessing.ipynb` for adding the silence class, checking split isolation, and extracting MFCC features. Run the new notebook locally after completing notebook 01.

The project uses the Google Speech Commands v0.02 dataset. The dataset contains 105,829 original one-second audio recordings across 35 spoken-word categories and background-noise recordings. The original dataset is distributed under the Creative Commons Attribution 4.0 licence.

- Dataset card: https://huggingface.co/datasets/google/speech_commands
- Original dataset archive: https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- Dataset paper: https://arxiv.org/abs/1804.03209

The initial classification target is 12 classes: ten core commands (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`), `unknown`, and `silence`. Auxiliary spoken words are grouped into `unknown`. One-second clips will be sampled from background-noise recordings to create `silence` examples.

## Repository structure

```text
notebooks/
  01_dataset_exploration.ipynb
src/
app/
api/
data/
  raw/
  processed/
models/
tests/
load_testing/
requirements.txt
```

## Local setup (Windows / Git Bash)

1. Clone the repository:

   ```bash
   git clone https://github.com/Abejirin-King/machine_learning_cycle.git
   cd machine_learning_cycle
   ```

2. Create and activate a virtual environment:

   ```bash
   py -3.11 -m venv .venv
   source .venv/Scripts/activate
   ```

   If the `py` launcher is unavailable, use `python -m venv .venv`.

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Run `notebooks/01_dataset_exploration.ipynb` first, then `notebooks/02_preprocessing.ipynb`.

Downloaded audio, generated indexes, and extracted feature arrays are local working data and must not be committed to GitHub.

## Current development environment

The local workflow uses Python directly and does not require Docker Desktop. Container-based deployment and scaling tests will be handled separately in a cloud environment if required by the assignment.

## Planned development

1. Dataset acquisition, split verification and exploratory analysis.
2. Audio preprocessing, MFCC extraction and spectrogram generation.
3. Traditional ML baselines and a CNN model.
4. Evaluation, error analysis and feature visualisations.
5. Single-file and bulk audio prediction.
6. Retraining pipeline and Python API.
7. Streamlit dashboard, cloud deployment and Locust load testing.

## Attribution

Speech Commands was created by Pete Warden and contributors. Please retain the dataset attribution and licence information when redistributing derived materials.
