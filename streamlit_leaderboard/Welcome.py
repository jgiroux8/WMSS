#from leaderboard_utils.utils import OPENAI_Utils, get_leaderboard_dataframe, log_out
#import json, sys, os
#from yaml.loader import SafeLoader
import streamlit as st
import pandas as pd
import sys 
from MNIST_Lecture import get_mnist_notebook
import fitz
from calendar_utils import get_calendar
from io import BytesIO

if "selected_field" not in st.session_state:
    st.session_state.selected_field = None

def display_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # open document
    for i, page in enumerate(doc):
        image = page.get_pixmap()  # render page to an image
        img_bytes = image.tobytes()
        img_buffer = BytesIO(img_bytes)
        st.image(img_buffer, caption=f"Slide {i+1}", use_column_width=True)


st.set_page_config(
    page_title="Open and FAIR Fusion for Maching Learning Applications",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://eic.ai',
        'About': "Three-year grant sponsored by the Department of Energy (DOE) Office of Fusion Energy Sciences (FES)."
    }
)



#st.image("/Users/james/AI4EICHackathon2023-Streamlit/WM_Logo.jpeg", caption=None,width=100)

def welcome_page():
    st.title("Open and FAIR Fusion for Maching Learning Applications")
    st.markdown("### Three-year grant sponsored by the Department of Energy (DOE) Office of Fusion Energy Sciences (FES).")
    st.divider()

    st.markdown("## A multi-institutional collaboration to develop a Fusion Data Platform for Machine Learning applications using Magnetic Fusion Energy (MFE) data.")
    st.markdown("MFE devices participating in this research are Alcator C-Mod, Pegasus-III, CTH and HBT-EP. An interoperable and publicly available library will be developed leveraging data from these devices. The library will have built-in pipelines for ML application design, allowing preservation of reproducible scientific results. Curated research products will be released through the newly designed platform, which will adhere to Findable, Accessible, Interoperable, Reusable (FAIR) and Open Science (OS) guidelines.")
    current_directory = os.getcwd()
    image_path1 = "WhoWeAre.png"
    image_path2 = "Affiliations.png"
    absolute_path1 = os.path.join(current_directory, image_path1)
    absolute_path2 = os.path.join(current_directory, image_path2)
    st.image(absolute_path1,use_column_width=True)
    st.image(absolute_path2,use_column_width=True)


    st.divider()
    st.markdown("## W&M Summer School")
    st.markdown("An intensive 2-week summer school focused on undergraduate students with backgrounds in physics, engineering, computer science, applied mathematics and data science will be offered at William & Mary. This summer course will include a close to equal distribution of traditional instruction and active projects. The traditional instruction will provide daily 50 min instruction in 4 classes with a focus on computing, applied mathematics, machine learning and fusion energy. These classes will be based on existing classes offered in data science at W&M, such as databases, applied machine learning, Bayesian reasoning in data science. These classes will be supplemented with a class focused on fusion energy for the applications the students will tackle during the hands-on component and for students’ summer research.")
    st.markdown("The 1st edition of the Summer School will be held in person at W&M during June 3-15, 2024. Open Call for registrations available soon & deadline end of January 2024.")
    st.markdown("[Register Here]({https://docs.google.com/forms/d/e/1FAIpQLScKUQtVPkZ4pDj2su3_xISVxHcXLdtuRpsGPJ4cJkK4BjkAXg/viewform})")
    st.markdown("**More details coming soon!**")

    st.divider()
    st.markdown("## External Colobarators")
    names = [
        "J. Levesque, Columbia University.",
        "N. Cummings, UKAEA.",
        "N. Murphy, Center for Astrophysics - Harvard & Smithsonian.",
        "A. Pau, EPFL SPC."
    ]
    bullet_points = "\n".join([f" * {name}" for name in names])
    st.markdown(bullet_points)
    st.markdown("and with the support of the International Atomic Energy Agency (IAEA).")

    st.divider()
    st.markdown("## In the News")

    links = [
    "[Fast-tracking fusion energy’s arrival with AI and accessibility](https://news.mit.edu/2023/fast-tracking-fusion-energy-with-ai-and-accessibility-0901)",
    "[William & Mary to lead machine learning efforts for nuclear fusion](https://news.wm.edu/2023/09/01/william-mary-to-lead-machine-learning-efforts-for-nuclear-fusion/)",
    "[UW–Madison part of effort to advance fusion energy with machine learning](https://news.wisc.edu/uw-madison-part-of-effort-to-advance-fusion-energy-with-machine-learning/)",
    "[FAIR data and inclusive science to enable clean energy](https://www.auburn.edu/cosam/news/articles/2023/08/fair_data_and_inclusive_science_to_enable_clean_energy.htm)",
    "[Department of Energy Awards Grant to The HDF Group and Collaborators for Fusion Energy Data Management Tools](https://www.hdfgroup.org/2023/09/department-of-energy-awards-grant-to-the-hdf-group-and-collaborators-for-fusion-energy-data-management-tools/)"]
    links = "\n".join([f" * {link}" for link in links])
    st.markdown(links)
    
def landing_page():
    st.title('Welcome to the W&M Summer School of 2024')
    st.divider()
    st.markdown('On the left sidebar, you can find a list of all relevant notebooks and lectures.')

def page_2024():
    st.title('Welcome to the W&M Summer School of 2024')
    st.divider()
    
    st.session_state.selected_field = st.sidebar.radio("Resources", ["Lecture 1", "Lecture 1 Notebook","Calendar"],index=None)
    
    if st.session_state.selected_field == "Lecture 1":
        image_path2 = "08_31_2022_Lec1_pub.pdf"
        pdf_path = os.path.join(current_directory, image_path1)
        display_pdf(pdf_path)
    elif st.session_state.selected_field == "Lecture 1 Notebook":
        get_mnist_notebook()
    elif st.session_state.selected_field == "Calendar":
        calendar = get_calendar()
        st.write(calendar)
            
        
def page_2025():
    st.title('Welcome to the W&M Summer School of 2025')
    st.divider()
    st.write('Under Development')
    

def main():
    # Set the default page to Welcome
    st.experimental_set_query_params(page="Welcome")

    # Check the URL parameters
    query_params = st.experimental_get_query_params()

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Welcome", "2024","2025"])

    if selected_page == "Welcome":
        welcome_page()
    elif selected_page == "2024":
        # Update URL to reflect the selected page
        st.experimental_set_query_params(page="2024")
        page_2024()
    elif selected_page == "2025":
        st.experimental_set_query_params(page="2025")
        page_2025()

if __name__ == "__main__":
    main()