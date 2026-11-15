"""Streamlit dashboard for the audio command recognition system."""
from __future__ import annotations
import io, json, os
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="Voice Command AI", page_icon="🎙️", layout="wide")

def api_get(base, path):
    r = requests.get(f"{base}{path}", timeout=60)
    r.raise_for_status()
    return r.json()

def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None

def show_features(audio_bytes, filename):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True, duration=1.0)
    if not audio.size:
        st.warning("No readable audio found.")
        return
    st.audio(audio_bytes, format="audio/wav")
    t = np.arange(len(audio)) / sr
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(t, audio, linewidth=0.7)
    ax.set(title=f"Waveform: {filename}", xlabel="Time (seconds)", ylabel="Amplitude")
    ax.grid(alpha=0.2)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=400, hop_length=160, n_mels=64)
    fig, ax = plt.subplots(figsize=(10, 3))
    image = librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=sr,
                                     hop_length=160, x_axis="time", y_axis="mel", ax=ax)
    ax.set(title="Log-Mel spectrogram")
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=400, hop_length=160)
    fig, ax = plt.subplots(figsize=(10, 3))
    image = librosa.display.specshow(mfcc, x_axis="time", ax=ax)
    ax.set(title="MFCC feature map", ylabel="MFCC coefficient")
    fig.colorbar(image, ax=ax)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

st.title("Voice Command Recognition")
st.caption("Audio classification dashboard | Google Speech Commands v0.02 | 12 classes")
with st.sidebar:
    st.header("API connection")
    api_base = st.text_input("FastAPI base URL", value=DEFAULT_API).strip().rstrip("/")
    st.caption("Start the API separately with uvicorn api.main:app --host 127.0.0.1 --port 8000")

health = metrics = None
try:
    health, metrics = api_get(api_base, "/health"), api_get(api_base, "/metrics")
except requests.RequestException as exc:
    st.error(f"Cannot reach API at {api_base}. Start FastAPI and check the URL. Details: {exc}")

if health and metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("API status", str(health.get("status", "unknown")).upper())
    c2.metric("Active model", metrics.get("active_model") or "Not loaded")
    c3.metric("Requests observed", metrics.get("requests_observed", 0))
    if not health.get("model_artifacts_ready", False):
        st.warning("API is running, but trained model artifacts are missing. Run notebooks 03–05 and place their generated files in models/.")
        st.write(health.get("missing_artifacts", []))
    else:
        st.success("Required model artifacts are ready.")

single_tab, batch_tab, feature_tab, performance_tab, retrain_tab = st.tabs(
    ["Single prediction", "Batch prediction", "Audio features", "Model performance", "Retrain"])

with single_tab:
    st.subheader("Predict one spoken command")
    audio = st.file_uploader("Upload a WAV recording", type=["wav"], key="single")
    if audio and st.button("Predict command", type="primary"):
        try:
            r = requests.post(f"{api_base}/predict",
                files={"file": (audio.name, audio.getvalue(), "audio/wav")}, timeout=120)
            r.raise_for_status()
            result = r.json()
            st.success(f"Predicted command: {str(result.get('predicted_command', '')).upper()}")
            a, b = st.columns(2)
            a.metric("Model", result.get("model", "unknown"))
            b.metric("Model score", f"{float(result.get('confidence_score', 0)):.4f}")
            st.caption(result.get("confidence_note", "Model score, not necessarily a calibrated probability."))
            if result.get("top_predictions"):
                st.dataframe(pd.DataFrame(result["top_predictions"]), use_container_width=True)
        except (requests.RequestException, ValueError, TypeError) as exc:
            st.error(f"Prediction failed: {exc}")

with batch_tab:
    st.subheader("Predict multiple recordings")
    uploads = st.file_uploader("Choose WAV files", type=["wav"], accept_multiple_files=True, key="batch")
    if uploads and st.button("Run batch prediction", type="primary"):
        files = [("files", (f.name, f.getvalue(), "audio/wav")) for f in uploads]
        try:
            r = requests.post(f"{api_base}/predict/batch", files=files, timeout=300)
            r.raise_for_status()
            frame = pd.DataFrame(r.json())
            st.dataframe(frame, use_container_width=True)
            st.download_button("Download predictions CSV", frame.to_csv(index=False),
                file_name="voice_command_predictions.csv", mime="text/csv")
            if "predicted_command" in frame:
                st.bar_chart(frame["predicted_command"].value_counts())
        except requests.RequestException as exc:
            st.error(f"Batch prediction failed: {exc}")

with feature_tab:
    st.subheader("Visualise audio features")
    feature_audio = st.file_uploader("Upload a WAV file", type=["wav"], key="features")
    if feature_audio:
        try:
            show_features(feature_audio.getvalue(), feature_audio.name)
            st.info("Waveform shows amplitude over time. The Mel spectrogram shows energy across perceptual frequency bands. MFCCs compress the spectral envelope into features used by the classical models.")
        except Exception as exc:
            st.error(f"Could not visualise recording: {exc}")

with performance_tab:
    st.subheader("Held-out test performance")
    performance = load_json(ROOT / "models" / "final_test_metrics.json")
    if performance is None:
        st.info("No local models/final_test_metrics.json found. Run notebook 05 and keep its generated metrics file in the models folder.")
    else:
        st.caption("These are saved held-out test results, not metrics recalculated by the dashboard.")
        rows = performance.get("test_metrics", [])
        if isinstance(rows, dict):
            rows = [{"model": name, **value} for name, value in rows.items() if isinstance(value, dict)]
        if isinstance(rows, list) and rows:
            frame = pd.DataFrame(rows)
            st.dataframe(frame, use_container_width=True)
            numeric = [c for c in ["accuracy", "balanced_accuracy", "macro_precision",
                "macro_recall", "macro_f1", "weighted_f1"] if c in frame.columns]
            if numeric and "model" in frame.columns:
                st.bar_chart(frame.set_index("model")[numeric])
        else:
            st.json(performance)
        st.write("Validation-selected model:", performance.get("validation_selected_model", "Not recorded"))
        st.caption("Validation metrics select the model. The held-out test split remains reserved for final evaluation.")

with retrain_tab:
    st.subheader("Submit newly labelled recordings")
    st.warning("Retraining is resource-intensive. Candidate promotion is gated by validation macro-F1. The held-out test set is never used.")
    st.markdown("Upload a CSV with columns filename,label. Each filename must match one uploaded WAV file.")
    metadata = st.file_uploader("Label metadata CSV", type=["csv"], key="labels")
    new_audio = st.file_uploader("New labelled WAV recordings", type=["wav"], accept_multiple_files=True, key="new_audio")
    if metadata:
        try:
            preview = pd.read_csv(metadata)
            st.dataframe(preview.head(10), use_container_width=True)
            metadata.seek(0)
            if not {"filename", "label"}.issubset(preview.columns):
                st.error("CSV must contain filename and label columns.")
        except Exception as exc:
            st.error(f"Could not read metadata CSV: {exc}")
    if metadata and new_audio and st.button("Start controlled retraining", type="primary"):
        files = [("metadata", (metadata.name, metadata.getvalue(), "text/csv"))]
        files.extend(("audio_files", (f.name, f.getvalue(), "audio/wav")) for f in new_audio)
        try:
            r = requests.post(f"{api_base}/retrain", files=files, timeout=1800)
            r.raise_for_status()
            result = r.json()
            if result.get("promoted"):
                st.success("Candidate passed the validation gate and was promoted.")
            else:
                st.info("Candidate did not beat the existing validation macro-F1. The deployed model was not changed.")
            st.json(result)
        except requests.RequestException as exc:
            st.error(f"Retraining failed: {exc}")

st.divider()
st.caption("Educational interface mapping: yes = confirm, no = reject, up/down = navigate, go = start or continue, stop = stop. Unknown and silence are detection classes.")
