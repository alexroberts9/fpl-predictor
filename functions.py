import pandas as pd
import requests


def get(item):
    response = requests.get("https://fantasy.premierleague.com/api/" + item + "/")
    data = response.json()
    return data


def current_gw():
    bootstrap = get("bootstrap-static")
    events = pd.DataFrame(bootstrap["events"])
    gw = events.loc[events["is_current"] == True, "id"].iloc[0]
    return gw


def user_team(entry_id):
    bootstrap = get("bootstrap-static")
    elements = pd.DataFrame(bootstrap["elements"])[["id", "web_name", "team", "element_type", "now_cost"]]
    elements["value"] = elements["now_cost"] / 10

    gw = current_gw()

    picks = get(f"entry/{entry_id}/event/{gw}/picks")
    picks_df = pd.DataFrame(picks["picks"])

    team_df = picks_df.merge(elements, left_on="element", right_on="id", how="left")

    team_df = team_df[[
        "element", "position", "web_name", "value", "multiplier",
        "is_captain", "is_vice_captain", "team"
    ]].sort_values("position")

    return team_df


def user_info(entry_id):
    entry_data = get(f"entry/{entry_id}")
    history_data = get(f"entry/{entry_id}/history")
    transfers_data = get(f"entry/{entry_id}/transfers")

    # basic team / manager info
    info = {
        "manager_name": f"{entry_data['player_first_name']} {entry_data['player_last_name']}",
        "team_name": entry_data["name"],
        "entry_id": entry_data["id"],
        "overall_points": entry_data["summary_overall_points"],
        "overall_rank": entry_data["summary_overall_rank"],
        "current_gw_points": entry_data["summary_event_points"],
        "current_gw_rank": entry_data["summary_event_rank"],
    }

    # season history by GW
    history_df = pd.DataFrame(history_data["current"])

    # past seasons, if you want them
    past_df = pd.DataFrame(history_data["past"])

    # chips played
    chips_df = pd.DataFrame(history_data["chips"])

    # transfer history
    transfers_df = pd.DataFrame(transfers_data)

    return info, history_df, past_df, chips_df, transfers_df
