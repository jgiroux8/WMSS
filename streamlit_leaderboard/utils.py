import streamlit as st
import fitz
from io import BytesIO

def scroll_top():
    js = '''
    <script>
        var body = window.parent.document.querySelector(".main");
        console.log(body);
        body.scrollTop = 0;
    </script>
    '''

    st.components.v1.html(js)

def display_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # open document
    for i, page in enumerate(doc):
        image = page.get_pixmap()  # render page to an image
        img_bytes = image.tobytes()
        img_buffer = BytesIO(img_bytes)
        st.image(img_buffer, caption=f"Slide {i+1}", use_column_width=True)
