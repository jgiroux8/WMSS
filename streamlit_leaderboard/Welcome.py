#from leaderboard_utils.utils import OPENAI_Utils, get_leaderboard_dataframe, log_out
#import json, sys, os
#from yaml.loader import SafeLoader
import streamlit as st
import pandas as pd
import sys
from MNIST_Lecture import get_mnist_notebook,initialize
from calendar_utils import get_calendar
import os
from PIL import Image
from utils import scroll_top,display_pdf

if "selected_field" not in st.session_state:
    st.session_state.selected_field = None


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

def welcome_page():

    st.markdown("# Open and FAIR Fusion for Maching Learning Applications")
    st.markdown("### Three-year grant sponsored by the Department of Energy (DOE) Office of Fusion Energy Sciences (FES).")
    st.divider()

    st.markdown("## A multi-institutional collaboration to develop a Fusion Data Platform for Machine Learning applications using Magnetic Fusion Energy (MFE) data.")
    st.markdown("MFE devices participating in this research are Alcator C-Mod, Pegasus-III, CTH and HBT-EP. An interoperable and publicly available library will be developed leveraging data from these devices. The library will have built-in pipelines for ML application design, allowing preservation of reproducible scientific results. Curated research products will be released through the newly designed platform, which will adhere to Findable, Accessible, Interoperable, Reusable (FAIR) and Open Science (OS) guidelines.")
    current_directory = os.getcwd()
    image_path1 = "WhoWeAre.png"
    image_path2 = "Affiliations.png"
    absolute_path1 = os.path.join(current_directory, image_path1)
    absolute_path2 = os.path.join(current_directory, image_path2)
    print(absolute_path1)
    st.image(Image.open(absolute_path1),use_column_width=True)
    st.image(Image.open(absolute_path2),use_column_width=True)


    st.divider()
    st.markdown("## ML/AI for Fusion Energy Summer School at W&M")
    st.markdown("An intensive 2-week summer school focused on undergraduate students with backgrounds in physics, engineering, computer science, applied mathematics and data science will be offered at William & Mary. This summer course will include a close to equal distribution of traditional instruction and active projects. The traditional instruction will provide daily 80 min instruction in 3 classes with a focus on computing, applied mathematics, machine learning and fusion energy. These classes will be based on existing classes offered in data science at W&M, such as databases, applied machine learning, Bayesian reasoning in data science. These classes will be supplemented with a class focused on fusion energy for the applications the students will tackle during the hands-on component and for students’ summer research. Stay tuned for a draft agenda to be posted soon!")
    st.markdown("The 1st edition of the Summer School will be held in person at W&M during June 3-15, 2024.")
    st.markdown("### Registrations are now open!")
    email_address = "wmsummerschool@gmail.com"
    gmail_link = f'<a href="mailto:{email_address}" target="_blank">wmsummerschool@gmail.com</a>'
    st.markdown(f'The [registration Google Form](https://docs.google.com/forms/d/e/1FAIpQLScbx-THJndD6zxPfOmg8PYbFZOO-M9SzFLRRJfCYHf2ZTstng/viewform?pli=1) requires you to sign-in with a Google account in order to upload your CV and unofficial transcript. For support and/or questions please write an email to {gmail_link}.', unsafe_allow_html=True)
    ai_image = os.path.join(current_directory,"AI_Thinking.png")
    left_co,cent_co,last_co = st.columns(3)
    with cent_co:
        st.image(Image.open(ai_image),width=400)
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

def page_2024():

    st.session_state.selected_field = st.sidebar.radio("Resources", ["Main Page","Calendar","Lecture 1", "Lecture 1 Notebook"])

    if st.session_state.selected_field == "Main Page":
        st.markdown("# Welcome to the W&M Summer School of 2024!")
        st.markdown("On the sidebar you can find selected links to various lectures and associated notebooks.")
        st.markdown("If you have any issues, feel free to contact us!")

    elif st.session_state.selected_field == "Lecture 1":
        top = st.session_state.top_button_clicked if "top_button_clicked" in st.session_state else False
        st.title("Lecture 1")
        st.divider()
        image_path3 = "08_31_2022_Lec1_pub.pdf"
        current_directory = os.getcwd()
        pdf_path = os.path.join(current_directory, image_path3)
        display_pdf(pdf_path)

        top = st.sidebar.button("Top of Page", key='top')
        if top:
            scroll_top()
            st.experimental_rerun()

    elif st.session_state.selected_field == "Lecture 1 Notebook":
        st.title('Example MNIST Notebook with Sklearn')
        st.divider()
        get_mnist_notebook()
    elif st.session_state.selected_field == "Calendar":
        st.title("Calendar")
        calendar = get_calendar()
        st.write(calendar)

def page_2025():
    st.title('Welcome to the W&M Summer School of 2025')
    st.divider()
    st.write('Under Development')


def main():
    # Set the default page to Welcome
    if "state" not in st.session_state:
        st.session_state.state = initialize()

    st.experimental_set_query_params(page="Welcome")

    # Check the URL parameters
    query_params = st.experimental_get_query_params()

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Welcome", "2024","2025"])
    if selected_page == "Welcome":
        welcome_page()
    elif selected_page == "2024":
        st.experimental_set_query_params(page="2024")
        page_2024()
    elif selected_page == "2025":
        st.experimental_set_query_params(page="2025")
        page_2025()


if __name__ == "__main__":
    main()
