import importlib.util
import inspect
import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="KaryoAI — AI Karyotype Analysis",
    page_icon="🧬",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "KaryoAI_refined.zip"


def load_pipeline():
    """Extract the current model bundle and discover a compatible inference function."""
    if not BUNDLE.exists():
        return None, "KaryoAI_refined.zip was not found in the repository root."

    extract_dir = Path(tempfile.gettempdir()) / "karyoai_runtime"
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".ready"

    try:
        if not marker.exists() or BUNDLE.stat().st_mtime > marker.stat().st_mtime:
            with zipfile.ZipFile(BUNDLE, "r") as zf:
                zf.extractall(extract_dir)
            marker.touch()

        candidates = []
        for py in extract_dir.rglob("*.py"):
            if py.name.startswith("_"):
                continue
            candidates.append(py)

        # Prefer likely inference modules.
        candidates.sort(key=lambda p: (not any(x in p.stem.lower() for x in ["infer", "predict", "pipeline", "karyo", "main"]), str(p)))

        for path in candidates:
            try:
                spec = importlib.util.spec_from_file_location(f"karyoai_{path.stem}", path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.path.insert(0, str(path.parent))
                spec.loader.exec_module(module)
                for fn_name in ["predict", "run_pipeline", "analyze", "process_image", "run_inference", "classify"]:
                    fn = getattr(module, fn_name, None)
                    if callable(fn):
                        return fn, f"Loaded {fn_name}() from {path.relative_to(extract_dir)}"
            except Exception:
                continue
    except Exception as exc:
        return None, f"Model bundle could not be loaded: {exc}"

    return None, "No compatible inference function was discovered. Add an inference function such as predict(image) to the model package."


st.title("🧬 KaryoAI")
st.caption("AI-assisted metaphase image analysis and karyotype workflow")

with st.sidebar:
    st.header("Analysis")
    st.write("Upload a Giemsa-stained or FISH metaphase image. The interface is designed to become the front end for the model in this repository.")
    mode = st.selectbox("Image type", ["Giemsa", "FISH", "Other"])
    show_debug = st.checkbox("Show model/debug information", value=False)

uploaded = st.file_uploader("Upload metaphase image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    left, right = st.columns(2)
    with left:
        st.subheader("Input image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("AI analysis")
        if st.button("Run KaryoAI", type="primary", use_container_width=True):
            with st.spinner("Running the model..."):
                fn, status = load_pipeline()
                if fn is None:
                    st.warning(status)
                    st.info("The web interface is installed, but the model's callable inference entry point still needs to be exposed. The app deliberately does not invent diagnostic results.")
                else:
                    try:
                        sig = inspect.signature(fn)
                        if len(sig.parameters) == 1:
                            result = fn(image)
                        else:
                            result = fn(image=image, mode=mode)
                        st.success("Analysis completed")
                        if isinstance(result, Image.Image):
                            st.image(result, use_container_width=True)
                        elif isinstance(result, dict):
                            st.json(result)
                        else:
                            st.write(result)
                    except Exception as exc:
                        st.error(f"Inference failed: {exc}")

        st.caption("Research use only — not a clinical diagnostic device.")

if show_debug:
    fn, status = load_pipeline()
    st.code(status)
