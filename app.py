from pathlib import Path
import importlib.util
import inspect
import io
import json
import sys
import tempfile
import zipfile

import streamlit as st
from PIL import Image

st.set_page_config(page_title="KaryoAI | Metaphase Analysis", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "KaryoAI_refined.zip"

st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 2rem;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.25); border-radius: 18px; margin-bottom: 1rem;}
.metric-card {padding: 1rem; border: 1px solid rgba(128,128,128,.2); border-radius: 14px;}
.small {font-size:.88rem; opacity:.75;}
</style>
""", unsafe_allow_html=True)


def load_pipeline():
    """Load an inference callable from the bundled KaryoAI model."""
    if not BUNDLE.exists():
        return None, "Model bundle not found: KaryoAI_refined.zip"
    runtime = Path(tempfile.gettempdir()) / "karyoai_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    marker = runtime / ".ready"
    try:
        if not marker.exists() or BUNDLE.stat().st_mtime > marker.stat().st_mtime:
            with zipfile.ZipFile(BUNDLE) as zf:
                zf.extractall(runtime)
            marker.touch()
        candidates = [p for p in runtime.rglob("*.py") if not p.name.startswith("_")]
        candidates.sort(key=lambda p: (not any(k in p.stem.lower() for k in ("infer", "predict", "pipeline", "karyo", "main")), str(p)))
        for path in candidates:
            try:
                spec = importlib.util.spec_from_file_location(f"karyoai_{path.stem}_{abs(hash(path))}", path)
                if not spec or not spec.loader:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.path.insert(0, str(path.parent))
                spec.loader.exec_module(module)
                for name in ("predict", "run_pipeline", "analyze", "process_image", "run_inference", "classify"):
                    fn = getattr(module, name, None)
                    if callable(fn):
                        return fn, f"{name}() • {path.relative_to(runtime)}"
            except Exception:
                continue
    except Exception as exc:
        return None, f"Model loading error: {exc}"
    return None, "No supported inference function was discovered in the model bundle."


def run_inference(fn, image, mode):
    sig = inspect.signature(fn)
    params = sig.parameters
    if len(params) == 1:
        return fn(image)
    kwargs = {}
    if "image" in params:
        kwargs["image"] = image
    elif "img" in params:
        kwargs["img"] = image
    else:
        return fn(image, mode) if len(params) >= 2 else fn(image)
    if "mode" in params:
        kwargs["mode"] = mode
    return fn(**kwargs)


st.markdown('<div class="hero"><h1>🧬 KaryoAI</h1><p>AI-assisted metaphase chromosome analysis for research and peer testing.</p><p class="small">Giemsa • FISH • Human & mouse metaphase workflows</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Analysis settings")
    mode = st.selectbox("Staining / image type", ["Giemsa", "FISH", "Other"])
    species = st.selectbox("Species", ["Human", "Mouse", "Unknown"])
    st.divider()
    show_debug = st.checkbox("Show technical diagnostics", False)
    st.caption("Research prototype — not clinically validated.")

uploaded = st.file_uploader("Upload a metaphase plate", type=["png", "jpg", "jpeg", "tif", "tiff"], help="Use a clear metaphase image. Original resolution is preferred.")

if not uploaded:
    st.info("Upload a metaphase image to begin. The app will display the image first and run the repository model when an inference entry point is available.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Workflow", "Metaphase → Karyotype")
    c2.metric("Stains", "Giemsa / FISH")
    c3.metric("Species", "Human / Mouse")
else:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read the image: {exc}")
        st.stop()

    st.subheader("1. Input")
    a, b = st.columns([1.35, 1])
    with a:
        st.image(image, caption=f"{uploaded.name} • {image.width} × {image.height}px", use_container_width=True)
    with b:
        st.write(f"**Species:** {species}")
        st.write(f"**Staining:** {mode}")
        st.write(f"**File:** {uploaded.name}")
        st.write(f"**Size:** {uploaded.size / 1024:.1f} KB")

    st.subheader("2. Run analysis")
    run = st.button("▶ Run KaryoAI", type="primary", use_container_width=True)
    if run:
        with st.spinner("Loading model and analysing metaphase image…"):
            fn, status = load_pipeline()
            if fn is None:
                st.warning("The interface is ready, but the current model bundle does not expose a supported inference function.")
                st.code(status)
                st.info("No chromosome calls or diagnostic findings are generated until the real model inference function is connected.")
            else:
                try:
                    result = run_inference(fn, image, mode)
                    st.success("Analysis completed")
                    st.subheader("3. Model output")
                    if isinstance(result, Image.Image):
                        st.image(result, use_container_width=True)
                    elif isinstance(result, dict):
                        st.json(result)
                        st.download_button("Download JSON result", json.dumps(result, indent=2, default=str), file_name="karyoai_result.json", mime="application/json")
                    elif isinstance(result, (str, int, float)):
                        st.write(result)
                    else:
                        st.write(result)
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")

st.divider()
st.markdown("**Research-use notice:** KaryoAI is a research prototype. Outputs must be independently reviewed and are not intended for clinical diagnosis or patient management.")

if show_debug:
    st.subheader("Technical diagnostics")
    _, status = load_pipeline()
    st.code(status)
    st.write({"bundle": str(BUNDLE), "bundle_exists": BUNDLE.exists(), "python": sys.version.split()[0]})
