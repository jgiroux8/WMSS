import streamlit as st
import os, json 
from leaderboard_utils.utils import clear_all


st.set_page_config(page_title="Logout", page_icon=":robot_face:", layout="wide", initial_sidebar_state="auto")
st.title("Welcome to AI4EIC Hackathon 2023")
st.divider()
st.header("Logout session")

if (not st.session_state.get("username") or not st.session_state.get("config")):
    st.info("You are already logged out of the session :stuck_out_tongue_winking_eye:")
    st.stop()


if  (st.session_state.get("username")):
    st.markdown(f"### Hi {st.session_state.NameOfUser} 🥷from {st.session_state.NameOfTeam} 👥")
    st.info("To log out press logout button below.. No confirmation will be shown")
    if(st.button("Logout")):
        cfg = st.session_state.get("config")
        cfg["status"] = "LOGGED_OUT"
        cfg["session_id"]+=1
        cfg["tokens_used"]+= st.session_state.get("total_tokens", 0)
        with open(os.path.join(st.session_state.userDir, "status.json"), 'w') as ifile:
            json.dump(cfg, ifile)
        clear_all()
        st.success("You have logged out successfully")


