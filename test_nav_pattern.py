import streamlit as st

pages = ["Home", "Modules", "EDA"]

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

def navigate_to(page):
    st.session_state.current_page = page

# Sidebar navigation
current_index = pages.index(st.session_state.current_page)
selected = st.sidebar.radio("Nav", pages, index=current_index, key=f"nav_{st.session_state.current_page}")
if selected != st.session_state.current_page:
    st.session_state.current_page = selected
    st.rerun()

st.write(f"Current page is: {st.session_state.current_page}")

if st.button("Go to Modules"):
    navigate_to("Modules")
    st.rerun()

if st.button("Go to EDA"):
    navigate_to("EDA")
    st.rerun()

if st.button("Go to Home"):
    navigate_to("Home")
    st.rerun()

