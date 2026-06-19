import streamlit as st
import requests
import pandas as pd

# TO RUN THE APP
# cd "/Users/alexroberts/Documents/Diss (new)/Stremlit"
# streamlit run app_v2.py

from functions import current_gw, user_team, user_info
from data_functions import (
    run_bayesian_lasso_pipeline,
    get_all_variables,
    generate_predictions
)


@st.cache_resource(show_spinner="Training model — this only runs once per server session...")
def load_model():
    return run_bayesian_lasso_pipeline()


@st.cache_data(ttl=3600, show_spinner="Fetching player data...")
def load_player_data():
    return get_all_variables()


st.title("FPL Predictor")

gw = current_gw()
st.write(f"Current Gameweek: {gw}")

beta_means, X_means, X_stds = load_model()
all_variables = load_player_data()

if "predictions" not in st.session_state:
    st.session_state.predictions = generate_predictions(all_variables, beta_means, X_means, X_stds)

predictions = st.session_state.predictions

# --- User team lookup ---
st.sidebar.header("Your FPL Team")
st.sidebar.caption("Find your team ID in the URL on the FPL website: fantasy.premierleague.com/entry/**123456**/event/38")

entry_id = st.sidebar.number_input("Enter your FPL team ID", min_value=1, step=1, value=None, placeholder="e.g. 123456")

if entry_id:
    try:
        info, history_df, _, _, _ = user_info(int(entry_id))
        st.sidebar.success(f"**{info['team_name']}** ({info['manager_name']})")
        st.sidebar.write(f"Overall points: {info['overall_points']}  |  Rank: {info['overall_rank']:,}")

        st.subheader(f"Your squad — GW{gw}")
        team_df = user_team(int(entry_id))
        st.dataframe(team_df, use_container_width=True)
    except Exception as e:
        st.sidebar.error(f"Could not load team — check the ID is correct. ({e})")
