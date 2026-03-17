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
    app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU for Streamlit Cloud
    swapper = insightface.model_zoo.get_model(
        "inswapper_128.onnx",
        download=True,
        download_zip=True
    )
    return app, swapper


app, swapper = load_models()


def swap_faces(src_pil, dest_pil):
    src_img = np.array(src_pil)
    dest_img = np.array(dest_pil)

    src_faces = app.get(src_img)
    dest_faces = app.get(dest_img)

    if len(src_faces) == 0 or len(dest_faces) == 0:
        raise ValueError("No faces detected in one of the images. Try clearer, front-facing photos.")

    source_face = src_faces[0]
    dest_face = dest_faces[0]

    result = swapper.get(dest_img, dest_face, source_face, paste_back=True)
    return Image.fromarray(np.uint8(result)).convert("RGB")


CUSTOM_CSS = """
<style>
.stApp {
  max-width: 1200px;
  margin: 0 auto;
}

.sticky-cta {
  position: sticky;
  top: 0;
  z-index: 999;
  background: #a5b4fc;
  color: #0f172a;
  padding: 10px 14px;
  text-align: center;
  border-bottom: 1px solid #333;
  display: block;
  text-decoration: none;
  border-radius: 10px;
  margin-bottom: 12px;
}
.sticky-cta:hover { filter: brightness(0.97); }
.sticky-cta .pill {
  background:#4f46e5;
  color:#fff;
  padding:4px 10px;
  border-radius:999px;
  margin-left:10px;
}

.api-cta-wrap {
  display:flex;
  justify-content:center;
  gap:12px;
  flex-wrap:wrap;
  margin: 14px 0 18px 0;
}
.api-cta-hero {
  display:inline-flex;
  align-items:center;
  gap:10px;
  padding:10px 14px;
  border-radius:14px;
  background: linear-gradient(90deg,#0ea5e9 0%, #a8a9de 100%);
  color:#fff !important;
  font-weight:800;
  letter-spacing:0.1px;
  box-shadow: 0 6px 22px rgba(99,102,241,0.35);
  border: 1px solid rgba(255,255,255,0.22);
  text-decoration:none !important;
}
.api-cta-hero:hover {
  filter:brightness(1.05);
  transform: translateY(-1px);
  transition: all .15s ease;
}
.api-cta-hero .new {
  background:#fff;
  color:#0ea5e9;
  font-weight:900;
  padding:2px 8px;
  border-radius:999px;
  font-size:12px;
  line-height:1;
}
.api-cta-hero .txt { font-weight:800; }
.api-cta-hero .chev { opacity:.95; }

.bottom-promo {
  position: fixed;
  left: 50%;
  transform: translateX(-50%);
  bottom: 16px;
  z-index: 1001;
  background:#0b0b0b;
  color:#fff;
  border: 1px solid #2a2a2a;
  border-radius: 12px;
  padding: 10px 14px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.bottom-promo a {
  color:#4ea1ff;
  text-decoration:none;
  font-weight:600;
}
.hero-center {
  text-align: center;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <a class="sticky-cta"
       href="https://www.face-swap.co/?utm_source=streamlit_faceswap&utm_medium=banner"
       target="_blank"
       rel="noopener">
       ⚡ <strong>Upgrade to HD</strong> — priority queue & higher resolution swaps!
       <span class="pill">GPU</span>
    </a>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-center">
      <h3>Image Face Swap (Preview)</h3>
      <p>
        <a href="https://www.face-swap.co/?utm_source=streamlit_faceswap&utm_medium=subtitle" target="_blank">
          face-swap.co
        </a>
      </p>
      <p>
        <strong>Free preview</strong> runs on CPU and may be limited in resolution to keep it fast.<br>
        Want full-quality <strong>HD AI face swap</strong> with GPU speed?
        <a href="https://www.face-swap.co/?utm_source=streamlit_faceswap&utm_medium=go_pro" target="_blank">
          Go Pro ↗
        </a>
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="api-cta-wrap">
      <a class="api-cta-hero"
         style="background: linear-gradient(90deg,#22c55e 0%, #16a34a 100%);"
         href="https://www.face-swap.co/?utm_source=streamlit_faceswap&utm_medium=hero_upgrade"
         target="_blank"
         rel="noopener">
        <span class="txt">⚡ Upgrade to HD (No code)</span>
        <span class="chev">↗</span>
      </a>

      <a class="api-cta-hero"
         href="https://www.face-swap.co/api?utm_source=streamlit_faceswap&utm_medium=hero_api_new"
         target="_blank"
         rel="noopener">
        <span class="new">DEV</span>
        <span class="txt">Face Swap API</span>
        <span class="chev">↗</span>
      </a>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown(
        """
### Upgrade to HD 1920x1080
- Higher resolution face swaps
- Priority queue
- API access & automation
- No watermark
"""
    )
    st.link_button(
        "Open Pro Checkout",
        "https://www.face-swap.co/?utm_source=streamlit_faceswap&utm_medium=sidebar_pro"
    )
    st.link_button(
        "API Access",
        "https://www.face-swap.co/api?utm_source=streamlit_faceswap&utm_medium=sidebar_api"
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
    st.link_button(
        "⚡ Upgrade to HD on face-swap.co",
        "https://www.face-swap.co/?utm_source=streamlit_faceswap&utm_medium=upgrade_to_hd",
        use_container_width=True
    )

with col2:
    st.subheader("Result")

    if run_button:
        st.warning(
            "Skip the limits — HD, priority queue & no watermark at face-swap.co"
        )

        if source_pil is None or target_pil is None:
            st.error("Please upload both a source and a target image.")
        else:
            try:
                with st.spinner("Swapping face..."):
                    out_img = swap_faces(source_pil, target_pil)

                st.image(out_img, caption="Result", use_container_width=True)
                st.success("Face swap complete.")
                st.info(
                    "✨ Like this preview? Get HD face swaps on face-swap.co."
                )

            except Exception as e:
                st.error(str(e))
    else:
        st.info("Result image will appear here.")

st.markdown(
    """
    <div class="bottom-promo">
      Want HD & faster processing?
      <a href="https://www.face-swap.co/?utm_source=streamlit_faceswap&utm_medium=upgrade" target="_blank">
        Upgrade
      </a>
    </div>
    """,
    unsafe_allow_html=True
)
