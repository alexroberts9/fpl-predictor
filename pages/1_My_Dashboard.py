import streamlit as st
import pandas as pd

from functions import get, current_gw, user_team, user_info

st.set_page_config(page_title="My FPL Dashboard", layout="wide")
st.title("My FPL Dashboard")

# --- Sidebar: entry ID (persisted across pages via session state) ---
st.sidebar.header("Your FPL Team")
st.sidebar.caption("Find your ID in the FPL URL: fantasy.premierleague.com/entry/**123456**/event/38")

default_id = st.session_state.get("entry_id", None)
entry_id = st.sidebar.number_input(
    "Enter your FPL team ID",
    min_value=1, step=1,
    value=default_id,
    placeholder="e.g. 123456"
)

if not entry_id:
    st.info("Enter your FPL team ID in the sidebar to get started.")
    st.stop()

st.session_state.entry_id = int(entry_id)
eid = int(entry_id)

try:
    entry_data = get(f"entry/{eid}")
    info, history_df, past_df, chips_df, transfers_df = user_info(eid)
    gw = current_gw()

    # player name/position lookup
    bootstrap = get("bootstrap-static")
    elements = pd.DataFrame(bootstrap["elements"])[["id", "web_name", "element_type"]]
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    id_to_name = dict(zip(elements["id"], elements["web_name"]))

    # ===================================================
    # 1. AT A GLANCE
    # ===================================================
    bank       = entry_data["last_deadline_bank"] / 10
    team_value = entry_data["last_deadline_value"] / 10

    st.subheader(f"{info['team_name']}  ·  {info['manager_name']}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Overall Points",  f"{info['overall_points']:,}")
    c2.metric("Overall Rank",    f"{info['overall_rank']:,}")
    c3.metric("GW Points",       info["current_gw_points"])
    c4.metric("GW Rank",         f"{info['current_gw_rank']:,}")
    c5.metric("Bank",            f"£{bank:.1f}m")
    c6.metric("Team Value",      f"£{team_value:.1f}m")

    st.divider()

    # ===================================================
    # 2. LEAGUES
    # ===================================================
    st.subheader("Leagues")
    classic_leagues = entry_data.get("leagues", {}).get("classic", [])

    if classic_leagues:
        rows = []
        for lg in classic_leagues:
            movement = lg["entry_last_rank"] - lg["entry_rank"]   # positive = risen
            if movement > 0:
                arrow = f"▲ {movement:,}"
            elif movement < 0:
                arrow = f"▼ {abs(movement):,}"
            else:
                arrow = "—"
            rows.append({
                "League":        lg["name"],
                "Rank":          f"{lg['entry_rank']:,}",
                "Last GW Rank":  f"{lg['entry_last_rank']:,}",
                "Movement":      arrow,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No leagues found.")

    st.divider()

    # ===================================================
    # 3. SEASON POINTS HISTORY
    # ===================================================
    st.subheader("Points per Gameweek")
    if not history_df.empty:
        chart_df = history_df.set_index("event")[["points", "total_points"]].rename(
            columns={"points": "GW Points", "total_points": "Cumulative Points"}
        )
        tab_gw, tab_cum = st.tabs(["GW by GW", "Cumulative"])
        with tab_gw:
            st.bar_chart(chart_df["GW Points"])
        with tab_cum:
            st.line_chart(chart_df["Cumulative Points"])

    st.divider()

    # ===================================================
    # 4. CHIPS STATUS
    # ===================================================
    st.subheader("Chips")
    all_chips = {
        "wildcard": "Wildcard",
        "freehit":  "Free Hit",
        "bboost":   "Bench Boost",
        "3xc":      "Triple Captain",
    }
    used_chips = set(chips_df["name"].tolist()) if not chips_df.empty else set()

    chip_cols = st.columns(4)
    for idx, (key, label) in enumerate(all_chips.items()):
        if key in used_chips:
            chip_cols[idx].metric(label, "Used ✓")
        else:
            chip_cols[idx].metric(label, "Available")

    st.divider()

    # ===================================================
    # 5. TRANSFER HISTORY
    # ===================================================
    st.subheader("Transfer History")
    if not transfers_df.empty:
        t = transfers_df.copy()
        t["Player In"]      = t["element_in"].map(id_to_name)
        t["Player Out"]     = t["element_out"].map(id_to_name)
        t["Cost In (£m)"]   = t["element_in_cost"] / 10
        t["Cost Out (£m)"]  = t["element_out_cost"] / 10
        t["GW"]             = t["event"]
        st.dataframe(
            t[["GW", "Player In", "Cost In (£m)", "Player Out", "Cost Out (£m)"]].sort_values("GW", ascending=False),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No transfers made yet.")

    st.divider()

    # ===================================================
    # 6. MODEL'S BEST PICKS
    # ===================================================
    st.subheader("Model's Best Picks")

    if "predictions" not in st.session_state:
        st.info("Go to the Home page first to run the model, then come back here for pick recommendations.")
    else:
        preds = st.session_state.predictions

        predicted_cols = sorted(
            [c for c in preds.columns if c.startswith("predicted_points_")],
            key=lambda c: int(c.split("_")[-1])
        )

        if predicted_cols:
            fixture_label = st.selectbox(
                "Show best picks for fixture:",
                options=predicted_cols,
                format_func=lambda c: f"Next fixture +{int(c.split('_')[-1]) - 1}" if int(c.split('_')[-1]) > 1 else "Next fixture"
            )

            top = (
                preds[["web_name_x", "position", "team_name", "value", fixture_label]]
                .dropna(subset=[fixture_label])
                .rename(columns={
                    "web_name_x":   "Player",
                    "team_name":    "Team",
                    "value":        "Price (£m)",
                    fixture_label:  "Predicted Pts",
                })
                .sort_values("Predicted Pts", ascending=False)
                .reset_index(drop=True)
            )
            top["Predicted Pts"] = top["Predicted Pts"].round(2)
            top.index += 1   # rank from 1

            tab_all, tab_gk, tab_def, tab_mid, tab_fwd = st.tabs(["Top 20", "GK", "DEF", "MID", "FWD"])
            with tab_all:
                st.dataframe(top.head(20), use_container_width=True)
            with tab_gk:
                st.dataframe(top[top["position"] == "GK"].head(10), use_container_width=True)
            with tab_def:
                st.dataframe(top[top["position"] == "DEF"].head(10), use_container_width=True)
            with tab_mid:
                st.dataframe(top[top["position"] == "MID"].head(10), use_container_width=True)
            with tab_fwd:
                st.dataframe(top[top["position"] == "FWD"].head(10), use_container_width=True)

            # --- Captain pick from the user's own squad ---
            st.subheader("Captain Recommendation")
            team_df = user_team(eid)
            squad_ids = set(team_df["element"].tolist())
            squad_preds = (
                preds[preds["id"].isin(squad_ids)][["id", "web_name_x", "position", fixture_label]]
                .dropna(subset=[fixture_label])
                .sort_values(fixture_label, ascending=False)
                .reset_index(drop=True)
            )

            if not squad_preds.empty:
                best = squad_preds.iloc[0]
                vc   = squad_preds.iloc[1] if len(squad_preds) > 1 else None

                col_c, col_vc = st.columns(2)
                col_c.success(f"**Captain: {best['web_name_x']}**  \nPredicted {best[fixture_label]:.2f} pts")
                if vc is not None:
                    col_vc.info(f"**Vice-Captain: {vc['web_name_x']}**  \nPredicted {vc[fixture_label]:.2f} pts")

                st.dataframe(
                    squad_preds.rename(columns={"web_name_x": "Player", fixture_label: "Predicted Pts"})[["Player", "position", "Predicted Pts"]],
                    use_container_width=True, hide_index=True
                )

except Exception as e:
    st.error(f"Could not load data — check your team ID is correct. ({e})")
