# Audio-Based Voice Command Recognition System

An end-to-end audio classification project that recognises short spoken commands and supports unknown-word and background-noise detection. The system is designed as a voice-controlled interface for an educational application.

## Project status

**Milestones 1-6: Dataset exploration, preprocessing, model training, held-out test evaluation, and single/batch prediction.** The repository includes `src/preprocessing.py`, `src/models.py`, `src/deep_learning.py`, `src/prediction.py`, and notebooks 01-06. Notebook 03 compares a class-weighted Linear SVM and Random Forest; notebook 04 trains a CNN on log-Mel spectrograms using all 12 classes, including silence. Notebook 05 evaluates all three trained models on the untouched test split and records the validation-selected model. Notebook 06 demonstrates single-file and batch WAV prediction using that selected model.

The project uses the Google Speech Commands v0.02 dataset. The dataset contains 105,829 original one-second audio recordings across 35 spoken-word categories and background-noise recordings. The original dataset is distributed under the Creative Commons Attribution 4.0 licence.

- Dataset card: https://huggingface.co/datasets/google/speech_commands
- Original dataset archive: https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- Dataset paper: https://arxiv.org/abs/1804.03209

The initial classification target is 12 classes: ten core commands (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`), `unknown`, and `silence`. Auxiliary spoken words are grouped into `unknown`. One-second clips will be sampled from background-noise recordings to create `silence` examples.

## Repository structure

```text
notebooks/
  01_dataset_exploration.ipynb
  02_preprocessing.ipynb
  03_model_training.ipynb
  04_cnn_training.ipynb
  05_final_test_evaluation.ipynb
  06_prediction_pipeline.ipynb
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

5. Run notebooks 01 through 05 in order. Notebook 04 uses PyTorch and torchaudio for CNN training. Run notebooks 03 and 04 again after pulling updates so both classical models and the CNN are saved with all 12 classes. Notebook 05 performs the final held-out test evaluation.
6. Run notebook 06 to test single-file and batch prediction. Put WAV files in `data/inference_samples/`; batch results are written to `outputs/batch_predictions.csv`.

Downloaded audio, generated indexes, and extracted feature arrays are local working data and must not be committed to GitHub.

## Current development environment

The local workflow uses Python directly and does not require Docker Desktop. Container-based deployment and scaling tests will be handled separately in a cloud environment if required by the assignment.

## Planned development

1. Dataset acquisition, split verification and exploratory analysis.
2. Audio preprocessing, MFCC extraction and spectrogram generation.
3. Traditional ML baselines and a CNN model (implemented in notebooks 03-04).
4. Validation comparison, final held-out test evaluation, error analysis and feature visualisations.
5. Single-file and bulk audio prediction (implemented in `src/prediction.py` and notebook 06).
6. Retraining pipeline and Python API.
7. Streamlit dashboard, cloud deployment and Locust load testing.

## Attribution

Speech Commands was created by Pete Warden and contributors. Please retain the dataset attribution and licence information when redistributing derived materials.
