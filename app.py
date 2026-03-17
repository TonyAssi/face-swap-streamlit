import os
import tempfile
from typing import Union, Any, Optional

import numpy as np
from PIL import Image
import streamlit as st
from gradio_client import Client, handle_file


ImageLike = Union[str, np.ndarray, Image.Image]

SPACE_ID = os.getenv("FACE_SWAP_SPACE_ID", "tonyassi/face-swap")
API_NAME = os.getenv("FACE_SWAP_API_NAME", "/swap_faces")


class FaceSwapClientError(Exception):
    pass


class InvalidImageError(FaceSwapClientError):
    pass


class RemoteInitError(FaceSwapClientError):
    pass


class RemoteCallError(FaceSwapClientError):
    pass


_CLIENT: Optional[Client] = None


@st.cache_resource
def get_client() -> Client:
    try:
        return Client(SPACE_ID, verbose=False)
    except Exception as e:
        raise RemoteInitError(
            f"Failed to initialize Gradio Client for Space '{SPACE_ID}'."
        ) from e


def _to_temp_png_path(img: ImageLike) -> str:
    if img is None:
        raise InvalidImageError("Image is None.")

    if isinstance(img, str):
        if not os.path.exists(img):
            raise InvalidImageError(f"File not found: {img}")
        return img

    if isinstance(img, Image.Image):
        pil = img.convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        pil.save(tmp.name, format="PNG")
        return tmp.name

    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise InvalidImageError("Numpy image must be HxWx3 or HxWx4.")

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.shape[2] == 4:
            arr = arr[:, :, :3]

        pil = Image.fromarray(arr).convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        pil.save(tmp.name, format="PNG")
        return tmp.name

    raise InvalidImageError(
        f"Unsupported image type: {type(img)}. Use str, PIL.Image, or numpy."
    )


def _cleanup_temp(path: str, original_input: ImageLike):
    try:
        if not isinstance(original_input, str) and path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _normalize_result_to_pil(result: Any) -> Image.Image:
    if isinstance(result, dict) and "path" in result and result["path"]:
        return Image.open(result["path"]).convert("RGB")

    if isinstance(result, str) and os.path.exists(result):
        return Image.open(result).convert("RGB")

    if isinstance(result, (list, tuple)) and result:
        return _normalize_result_to_pil(result[0])

    if isinstance(result, Image.Image):
        return result.convert("RGB")

    raise RemoteCallError(f"Unexpected result type from Space: {type(result)}")


def swap_faces(src_img: ImageLike, dest_img: ImageLike) -> Image.Image:
    client = get_client()

    src_path = _to_temp_png_path(src_img)
    dest_path = _to_temp_png_path(dest_img)

    try:
        result = client.predict(
            src_img=handle_file(src_path),
            dest_img=handle_file(dest_path),
            api_name=API_NAME,
        )
        return _normalize_result_to_pil(result)

    except Exception as e:
        raise RemoteCallError(
            f"Remote face swap call failed for Space '{SPACE_ID}' "
            f"with api_name '{API_NAME}'."
        ) from e

    finally:
        _cleanup_temp(src_path, src_img)
        _cleanup_temp(dest_path, dest_img)


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
  background: linear-gradient(90deg,#22c55e 0%, #16a34a 100%);
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

st.set_page_config(page_title="Image Face Swap", layout="wide")
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
        <strong>Free preview</strong> runs on shared inference via Hugging Face.<br>
        Want full-quality <strong>HD AI face swap</strong> with better reliability?
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
         href="https://www.face-swap.co/?utm_source=streamlit_faceswap&utm_medium=hero_upgrade"
         target="_blank"
         rel="noopener">
        <span class="txt">⚡ Upgrade to HD (No code)</span>
        <span class="chev">↗</span>
      </a>
    </div>
    """,
    unsafe_allow_html=True
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
    if source_image is not None:
        source_pil = Image.open(source_image).convert("RGB")
        st.image(source_pil, caption="Source Image", use_container_width=True)

    target_image = st.file_uploader(
        "Target Image (face to replace)",
        type=["png", "jpg", "jpeg", "webp"],
        key="dst"
    )
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
        st.warning("Skip the limits — HD, priority queue & no watermark at face-swap.co")

        if source_pil is None or target_pil is None:
            st.error("Please upload both a source and a target image.")
        else:
            try:
                with st.spinner("Swapping face..."):
                    out_img = swap_faces(source_pil, target_pil)

                st.image(out_img, caption="Result", use_container_width=True)
                st.success("Face swap complete.")
                st.info("✨ Like this preview? Get HD face swaps on face-swap.co.")

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
