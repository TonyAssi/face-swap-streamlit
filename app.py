import streamlit as st
from PIL import Image
import numpy as np
import insightface
from insightface.app import FaceAnalysis

st.set_page_config(page_title="Image Face Swap", layout="wide")

assert insightface.__version__ >= "0.7"

@st.cache_resource
def load_models():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU
    swapper = insightface.model_zoo.get_model(
        "inswapper_128.onnx",
        download=True,
        download_zip=True
    )
    return app, swapper

app, swapper = load_models()

st.title("Image Face Swap (Preview)")
st.markdown(
    """
    [face-swap.co](https://www.face-swap.co/?utm_source=streamlit_faceswap&utm_medium=subtitle)

    **Free preview** runs on CPU and may be limited in resolution to keep it fast.  
    Want full-quality **HD AI face swap** with GPU speed?  
    [Go Pro ↗](https://www.face-swap.co/?utm_source=streamlit_faceswap&utm_medium=go_pro)
    """
)

col1, col2 = st.columns(2)

source_pil = None
target_pil = None

with col1:
    source_image = st.file_uploader(
        "Source Image (face to copy)",
        type=["png", "jpg", "jpeg", "webp"],
        key="src"
    )
    target_image = st.file_uploader(
        "Target Image (face to replace)",
        type=["png", "jpg", "jpeg", "webp"],
        key="dst"
    )

    if source_image is not None:
        source_pil = Image.open(source_image).convert("RGB")
        st.image(source_pil, caption="Source Image", use_container_width=True)

    if target_image is not None:
        target_pil = Image.open(target_image).convert("RGB")
        st.image(target_pil, caption="Target Image", use_container_width=True)

    run_button = st.button("Swap Face", type="primary")

with col2:
    st.subheader("Result")
    st.info("Result image will appear here.")

if run_button:
    if source_pil is None or target_pil is None:
        st.error("Please upload both a source and a target image.")
    else:
        st.success("Models loaded correctly. Next we’ll add the swap logic.")
