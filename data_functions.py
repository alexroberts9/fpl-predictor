import pandas as pd
import numpy as np
from statistics import mean
from scipy.stats import invgamma
# for api
import json
import requests

#Define Gibbs sampler
# X is the predictor variables df
# y is the target df

def gibbs_bayesian_lasso(X, y, num_iter, lambda_, a_sigma, b_sigma,
                        init_beta, init_sigma2, init_tau2):

    n, p = X.shape

    # standardise X
    X_mean = X.mean()
    X_std  = X.std().replace(0, 1)   # replace 0 std to avoid division by zero
    X = (X - X_mean) / X_std

    #normalise y
    y_tilda = y - mean(y)

    # define parameter variables
    beta = [0]*p if init_beta is None else init_beta
    sigma2 = init_sigma2
    tau2 = [1]*p if init_tau2 is None else init_tau2
    beta_samples = np.zeros((num_iter, p))


    # precompute matrix calculations
    # '@' is matrix multiplication in python
    # 't' means transpose in my definitions
    XtX = X.T @ X
    Xt_ytilda = X.T @ y_tilda


    # Define a min for tau to avoid zero division error and also needs not to be too close to 0
    eps = 1e-12


    # loop each iteration
    for i in range(num_iter):

        # ======================================================
        # 1. beta | tau2, sigma2, y_tilda
        # ======================================================

        # D^{-1} as pandas diagonal DataFrame
        Dinv = pd.DataFrame(
            np.diag(1.0 / np.clip(tau2, eps, None)), # this places a min on tau so cant be too close to 0
            index=X.columns, # just names the columns and rows
            columns=X.columns
        )

        A = XtX + Dinv

        # define mean and covariance parameters for our Normal distribution
        beta_mean = np.linalg.solve(A, Xt_ytilda)
        covariance_matrix = sigma2 * np.linalg.inv(A)

        # define beta
        beta = np.random.multivariate_normal(
            mean=beta_mean.flatten(),     # make it 1D
            cov=covariance_matrix)


        # ======================================================
        # 2. tau2 | beta, sigma2
        # ======================================================

        # 1/τj has Inverse gausian distribution, with parameter mu and scale(/shape) below
        # 1/τj^2 | βj,σ2 ∼ Inv-Gaussian( sqrt((λ^2 * σ^2)/β^2), λ^2)

        for j in range(p):
            beta_j_squared = max(beta[j]**2, eps)
            mu_tau = np.sqrt((lambda_**2 * sigma2) / beta_j_squared)

            # note that wald distribution is inverse gaussian
            inv_tau2 = np.random.wald(mu_tau, lambda_**2)
            tau2[j] = 1.0 / max(inv_tau2, eps)


        # ======================================================
        # 3. sigma2 | beta, tau2, y
        # ======================================================

        # we place a non-zero prior on the two parameters of the inverse gamma dist of sigma squared, a_sigma and b_sigma
        # using small positive values gives a proper posterior as oppsed to using (0,0) and dont make a difference

        ### precomputing of values
        # residual = y_tilda - X beta
        residual = y_tilda - X @ beta
        # first quadratic form: (y - Xβ)^T (y - Xβ)
        term1 = residual @ residual
        # second quadratic form: β^T D^{-1} β - this is still correct and quicker
        term2 = np.sum((beta**2) / np.clip(tau2, eps, None))


        shape = a_sigma + (((n - 1) + p)/2)
        scale = b_sigma + ((term1 + term2)/2)

        sigma2 = invgamma.rvs(a=shape, scale=scale)


        # ======================================================
        # 4. add current iteration beta sample, beta[i], to total list
        # ======================================================

        beta_samples[i] = beta


    return beta_samples, X_mean, X_std


# Define Optimise lambda function — uses held-out test set and R² instead of training RSS
def optimise_lambda(X_train, y_train, X_test, y_test, lambda_grid):
    best_lambda = None
    best_r2 = -np.inf

    for lambda_ in lambda_grid:
        beta_samples, X_mean, X_std = gibbs_bayesian_lasso(
            X_train, y_train, num_iter=1000, lambda_=lambda_,
            a_sigma=0.01, b_sigma=0.01, init_beta=None, init_sigma2=1, init_tau2=None)

        burn_in = 250
        beta_mean = beta_samples[burn_in:].mean(axis=0)

        y_mean = np.mean(y_train)
        X_test_scaled = (X_test - X_mean) / X_std
        predictions = X_test_scaled @ beta_mean + y_mean
        residuals = y_test - predictions

        SS_res = np.sum(residuals ** 2)
        SS_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - SS_res / SS_tot

        if r2 > best_r2:
            best_r2 = r2
            best_lambda = lambda_

    return best_lambda


 # Combine data function
def combine_position_data(pos, start_week, end_week):
    combined_data = []

    base_path = "/Users/alexroberts/Documents/Diss (new)/(1) Making csv/Individual GW+position files"


    for i in range(start_week, end_week + 1):
        file_name = f"{pos}_GW_{i}_players.csv"
        file_path = f"{base_path}/{file_name}"

        try:
            data = pd.read_csv(file_path)
            data["gameweek"] = i
            combined_data.append(data)
        except FileNotFoundError:
            pass

    if combined_data:
        combined_data = pd.concat(combined_data, ignore_index=True)
        combined_data = combined_data.dropna()
    else:
        combined_data = pd.DataFrame()

    return combined_data



def run_bayesian_lasso_pipeline():

    positions = ["GK", "DEF", "MID", "FWD"]

    rolling_columns = [
        "team", "opponent_team", "value", "was_home",
        "ewma_total_points", "ewma_influence", "ewma_creativity", "ewma_threat",
        "avg_total_points_20", "avg_influence_20", "avg_creativity_20", "avg_threat_20"
    ]

    # Store optimal lambda for each position
    optimal_lambdas = {}

    # Loop through each position — use 21/22/23 seasons for train, 24 for test
    for pos in positions:
        train_data = combine_position_data(pos, 1, 114)
        test_data  = combine_position_data(pos, 115, 152)

        y_train = train_data["total_points"]
        y_test  = test_data["total_points"]

        X_train = train_data[rolling_columns]
        X_test  = test_data[rolling_columns]

        lambda_grid = np.arange(0.01, 2.51, 0.25)
        optimal_lambda = optimise_lambda(X_train, y_train, X_test, y_test, lambda_grid)

        optimal_lambdas[pos] = optimal_lambda


    ###########################
    # PERFORMANCE TESTING & PLOTTING (with 3 chains)

    X_means = {}
    X_stds  = {}

    beta_means = []

    for pos in positions:

        # Combine all data for the position
        train_data = combine_position_data(pos, 1, 152)

        y_train = train_data["total_points"]

        predictor_variables = [
            "team", "opponent_team", "value", "was_home",
            "ewma_total_points", "ewma_influence", "ewma_creativity", "ewma_threat",
            "avg_total_points_20", "avg_influence_20",
            "avg_creativity_20", "avg_threat_20"
        ]

        X_train = train_data[rolling_columns]

        X_means[pos] = X_train.mean()
        X_stds[pos]  = X_train.std().replace(0, 1)

        _, p = X_train.shape

        optimal_lambda = optimal_lambdas[pos]

        #####################
        # Run 3 chains

        # Chain 1: default
        chain1, _, _ = gibbs_bayesian_lasso(
            X_train, y_train,
            lambda_=optimal_lambda,
            init_beta=np.zeros(p),
            init_sigma2=1.0,
            init_tau2=np.ones(p), num_iter=1000, a_sigma=0.01, b_sigma=0.01
        )
        chain1_df = pd.DataFrame(chain1, columns=rolling_columns)
        chain1_df["Chain"] = "Chain 1"

        # Chain 2: random start
        chain2, _, _ = gibbs_bayesian_lasso(
            X_train, y_train,
            lambda_=optimal_lambda,
            init_beta=np.random.uniform(-5, 5, p),
            init_sigma2=2.0,
            init_tau2=np.random.uniform(0.5, 2, p), num_iter=1000, a_sigma=0.01, b_sigma=0.01
        )
        chain2_df = pd.DataFrame(chain2, columns=rolling_columns)
        chain2_df["Chain"] = "Chain 2"

        # Chain 3: large start
        chain3, _, _ = gibbs_bayesian_lasso(
            X_train, y_train,
            lambda_=optimal_lambda,
            init_beta=np.full(p, 10.0),
            init_sigma2=5.0,
            init_tau2=np.full(p, 5.0), num_iter=1000, a_sigma=0.01, b_sigma=0.01
        )
        chain3_df = pd.DataFrame(chain3, columns=rolling_columns)
        chain3_df["Chain"] = "Chain 3"


        burn_in = 250

        combined_samples = pd.concat([
            chain1_df.loc[burn_in:, predictor_variables],
            chain2_df.loc[burn_in:, predictor_variables],
            chain3_df.loc[burn_in:, predictor_variables]
        ])

        beta_mean = combined_samples.mean(axis=0)
        beta_means.append((beta_mean, pos))


    return beta_means, X_means, X_stds




# MOVING ONTO API DATA


def get(item):
    response = requests.get('https://fantasy.premierleague.com/api/'+item+'/')
    data = response.json()
    return data


# creates one df which has each players team and position
def get_general_player_df(general_info):
    # player info
    players = pd.json_normalize(general_info['elements'])
    player_info = players[['id', 'web_name', 'first_name', 'second_name', 'team', 'element_type']].copy()

    # teams info
    teams = pd.json_normalize(general_info['teams'])
    teams = teams[['id', 'name']].copy()

    # positions info
    positions = pd.json_normalize(general_info['element_types'])
    positions = positions[['id', 'plural_name_short']].copy()

    # combining
    general_player_info = pd.merge(player_info, positions, left_on='element_type', right_on='id', how='left')
    general_player_info = pd.merge(player_info, teams, left_on='team', right_on='id')

    # editing columns
    general_player_info = general_player_info.drop(columns=['team', 'element_type', 'id_y'])
    general_player_info = general_player_info.rename(columns={'id_x':'id', 'name':'team'})

    return general_player_info


# for a given player this calcualtes all the past variables
def individual_player_predictor_df(player, general_player_df, decay_span, rolling_window):
    # player is just the players id
    # general_player_df just contains their name and their team
    first_name = general_player_df.loc[general_player_df['id']==player, 'first_name'].item()
    second_name = general_player_df.loc[general_player_df['id']==player, 'second_name'].item()
    name = f'{first_name} {second_name}'

    #### get gw info for each player
    # this brings me up info of the player
    player_df = get(f'element-summary/{player}')
    # this gets the past part of the player df, remeber the keys for 'get(f'element-summary/{player}')' are 'fixtures',
    #'hisotry' (this seaon psat gw), and 'history past' (previous season data)
    player_df = pd.json_normalize(player_df['history'])
    # columns for history include ['element', 'fixture', 'opponent_team', 'total_points', 'was_home',
        #'kickoff_time', 'team_h_score', 'team_a_score', 'round', 'modified',
       #'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
       #'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
       #'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity',
       #'threat', 'ict_index', 'clearances_blocks_interceptions', 'recoveries',
       #'tackles', 'defensive_contribution', 'starts', 'expected_goals',
       #'expected_assists', 'expected_goal_involvements',
       #'expected_goals_conceded', 'value', 'transfers_balance', 'selected',
       #'transfers_in', 'transfers_out']
    player_df = player_df[['element', 'total_points', 'influence', 'creativity', 'threat']]

    # combine past gw info with general info (ie combine name and team with past gw df)
    final_df = pd.merge(player_df, general_player_df, left_on='element', right_on='id')
    # editing opponent team from id to name - this is y we need teams df
    player_df = final_df


    # make extra variables
    # Calculate EWMA and rolling averages
    player_df['ewma_total_points'] = player_df['total_points'].shift(1).ewm(span=decay_span, adjust=False).mean().round(2)
    player_df['ewma_influence'] = player_df['influence'].shift(1).ewm(span=decay_span, adjust=False).mean().round(2)
    player_df['ewma_creativity'] = player_df['creativity'].shift(1).ewm(span=decay_span, adjust=False).mean().round(2)
    player_df['ewma_threat'] = player_df['threat'].shift(1).ewm(span=decay_span, adjust=False).mean().round(2)

    player_df['avg_total_points_20'] = player_df['total_points'].shift(1).rolling(window=rolling_window, min_periods=1).mean().round(2)
    player_df['avg_influence_20'] = player_df['influence'].shift(1).rolling(window=rolling_window, min_periods=1).mean().round(2)
    player_df['avg_creativity_20'] = player_df['creativity'].shift(1).rolling(window=rolling_window, min_periods=1).mean().round(2)
    player_df['avg_threat_20'] = player_df['threat'].shift(1).rolling(window=rolling_window, min_periods=1).mean().round(2)

    player_df['Full name'] = name
    # remove non-variable colums
    cols = ['id', 'Full name',
            'ewma_total_points', 'ewma_influence', 'ewma_creativity', 'ewma_threat',
            'avg_total_points_20', 'avg_influence_20', 'avg_creativity_20', 'avg_threat_20']

    player_df = player_df[cols]
    # return only last weekl
    final_week = player_df.iloc[[-1]].copy()

    return final_week


# This applies the prevuous function to every player, outputting the past variables for every player
def get_player_predictor_variables_df():
    decay_span = 5
    rolling_window = 20

    general_info = get('bootstrap-static')
    # get general name and team info
    general_player_df = get_general_player_df(general_info)
    # list of player
    players = general_player_df['id']
    all_player_dfs = []

    # iterate through each player to get the past variables for the next fixture
    for player in players:
        past_variables = individual_player_predictor_df(player, general_player_df, decay_span, rolling_window)
        all_player_dfs.append(past_variables)

    all_player_df = pd.concat(all_player_dfs, ignore_index=True)

    return all_player_df



def get_present_variables():
    ### presnet info - current team and value
    general_info = get("bootstrap-static")

    players = pd.DataFrame(general_info["elements"])
    teams = pd.DataFrame(general_info["teams"])[["id", "short_name"]]

    # Convert price to £ format
    players["value"] = players["now_cost"] / 10

    # Map team ID → team name
    id_to_team = dict(zip(teams["id"], teams["short_name"]))
    players["team_name"] = players["team"].map(id_to_team)

    # Keep only what you need
    current_info = players[[
        "id",
        "web_name",
        "team_name",
        "value"
    ]]

    return current_info



def get_future_variables():

    team_rank_2024_25_FLIPPED = {
        # Promoted
        "SUN": 20,
        "BUR": 19,
        "LEE": 18,

        # 2024/25 PL teams
        "TOT": 17,
        "WOL": 16,
        "MUN": 15,
        "WHU": 14,
        "EVE": 13,
        "CRY": 12,
        "FUL": 11,
        "BRE": 10,
        "BOU": 9,
        "BHA": 8,
        "NFO": 7,
        "AVL": 6,
        "NEW": 5,
        "CHE": 4,
        "MCI": 3,
        "ARS": 2,
        "LIV": 1
    }

    # Fixtures (future only)
    fixtures = pd.DataFrame(get("fixtures"))[["event", "team_h", "team_a", "finished"]]
    future = fixtures[fixtures["finished"] == False].copy()

    # Team-centric rows + venue
    home = (
        future.rename(columns={"team_h": "team", "team_a": "opponent"})[["event", "team", "opponent"]]
        .assign(is_home=1)
    )

    away = (
        future.rename(columns={"team_a": "team", "team_h": "opponent"})[["event", "team", "opponent"]]
        .assign(is_home=0)
    )

    team_fixtures = pd.concat([home, away], ignore_index=True).sort_values("event", na_position="last")

    # Players + teams
    bootstrap = get("bootstrap-static")
    players = pd.DataFrame(bootstrap["elements"])[["id", "web_name", "team"]]
    teams = pd.DataFrame(bootstrap["teams"])[["id", "short_name"]]
    id_to_team = dict(zip(teams["id"], teams["short_name"]))

    # Add player's own team short name and rank
    players["team_name"] = players["team"].map(id_to_team)
    players["team"] = players["team_name"].map(team_rank_2024_25_FLIPPED)

    # Merge fixtures onto players
    players_fixtures = players.merge(team_fixtures, on="team", how="left")
    players_fixtures["Opponent"] = players_fixtures["opponent"].map(id_to_team)
    players_fixtures["event"] = players_fixtures["event"].astype("Int64")

    # Map opponent short_name -> opponent rank
    players_fixtures["opp_rank"] = players_fixtures["Opponent"].map(team_rank_2024_25_FLIPPED)

    # Keep tidy
    final_fixtures = players_fixtures[["id", "web_name", "team", "event", "opp_rank", "is_home"]].copy()

    # Sort within player, create fixture number 1..8, keep first 8
    final_fixtures = final_fixtures.sort_values(["web_name", "event"], na_position="last")
    final_fixtures["fixture_no"] = final_fixtures.groupby("web_name").cumcount() + 1
    final_fixtures = final_fixtures[final_fixtures["fixture_no"] <= 8]

    # Pivot to wide: opp_rank columns
    opp = (
        final_fixtures.pivot(index="web_name", columns="fixture_no", values="opp_rank")
        .rename(columns=lambda k: f"opp_rank_{k}")
        .reset_index()
    )

    # Pivot to wide: is_home columns
    homeaway = (
        final_fixtures.pivot(index="web_name", columns="fixture_no", values="is_home")
        .rename(columns=lambda k: f"is_home_{k}")
        .reset_index()
    )

    # Keep one team rank per player
    team_rank = final_fixtures[["web_name", "team"]].drop_duplicates()

    # Merge pivots + keep id too
    ids = players[["web_name", "id"]].drop_duplicates()
    df = (
        ids.merge(team_rank, on="web_name", how="left")
          .merge(opp, on="web_name", how="left")
          .merge(homeaway, on="web_name", how="left")
    )

    return df



def get_all_variables():

    past_variables_df = get_player_predictor_variables_df()
    present_variables = get_present_variables()
    future_variables = get_future_variables()

    all_variables = pd.merge(past_variables_df, present_variables, on="id")
    all_variables = pd.merge(all_variables, future_variables, on="id")
    all_variables = all_variables.drop(columns=['web_name_y'])

    return all_variables



def generate_predictions(all_variables, beta_means, X_means, X_stds):

    rolling_columns = [
        "team", "opponent_team", "value", "was_home",
        "ewma_total_points", "ewma_influence", "ewma_creativity", "ewma_threat",
        "avg_total_points_20", "avg_influence_20", "avg_creativity_20", "avg_threat_20"
    ]

    beta_dict = {}
    for beta, pos in beta_means:
        beta_dict[pos] = beta

    predictions = all_variables.copy()

    # Add position internally
    bootstrap = get("bootstrap-static")
    player_positions = pd.DataFrame(bootstrap["elements"])[["id", "element_type"]]

    pos_map = {
        1: "GK",
        2: "DEF",
        3: "MID",
        4: "FWD"
    }

    player_positions["position"] = player_positions["element_type"].map(pos_map)
    player_positions = player_positions[["id", "position"]]

    predictions = predictions.merge(player_positions, on="id", how="left")

    # Use whichever team-strength column exists
    if "team_rank" in predictions.columns:
        team_col = "team_rank"
    elif "team" in predictions.columns:
        team_col = "team"
    else:
        raise KeyError("Neither 'team_rank' nor 'team' exists in all_variables")

    for i in range(1, 9):

        if f"opp_rank_{i}" not in predictions.columns:
            break

        X = pd.DataFrame({
            "team": predictions[team_col],
            "opponent_team": predictions[f"opp_rank_{i}"],
            "value": predictions["value"],
            "was_home": predictions[f"is_home_{i}"],
            "ewma_total_points": predictions["ewma_total_points"],
            "ewma_influence": predictions["ewma_influence"],
            "ewma_creativity": predictions["ewma_creativity"],
            "ewma_threat": predictions["ewma_threat"],
            "avg_total_points_20": predictions["avg_total_points_20"],
            "avg_influence_20": predictions["avg_influence_20"],
            "avg_creativity_20": predictions["avg_creativity_20"],
            "avg_threat_20": predictions["avg_threat_20"]
        })

        X = X[rolling_columns]
        predictions[f"predicted_points_{i}"] = np.nan

        for pos in beta_dict:
            mask = predictions["position"] == pos
            # apply the same standardisation used during training
            X_scaled = (X.loc[mask] - X_means[pos]) / X_stds[pos]
            predictions.loc[mask, f"predicted_points_{i}"] = (
                X_scaled @ beta_dict[pos]
            )

    return predictions
