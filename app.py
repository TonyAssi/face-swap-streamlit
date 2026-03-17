import streamlit as st

st.set_page_config(page_title="Image Face Swap", layout="wide")

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
    run_button = st.button("Swap Face", type="primary")

with col2:
    st.subheader("Result")
    st.info("Result image will appear here.")

if run_button:
    st.success("UI is wired. Next we’ll connect the face swap logic.")
