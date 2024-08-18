import streamlit as st
import pandas as pd
from sklearn import tree

#Path
path = "./data/modelling.csv"

#Load data
@st.cache_data
def load_data():
    df = pd.read_csv(path)
    df = df.drop("Unnamed: 0", axis = 1)
    return df

#Load data, copy df, convert Year to str
data = load_data()
data = data.rename(columns = {"off_total_yds_g":"Offensive Total Yds. Per Game",
                                    "off_pass_yds_g":"Offensive Passing Yds. Per Game", 
                                    "off_rush_yds_g":"Offensive Rushing Yds. Per Game", 
                                    "off_pts_g":"Offensive Points Per Game", 
                                    "def_total_yds_g":"Defensive Total Yds. Per Game", 
                                    "def_pass_yds_g":"Defensive Passing Yards Per Game", 
                                    "def_rush_yds_g": "Defensive Rushing Yards Per Game",
                                    "def_pts_g":"Defensive Points Per Game",
                                    "kick_att":"Kick Returns",
                                    "kick_yds":"Kick Return Yds.",
                                    "kick_avg":"Yds. Per Kick Return",
                                    "kick_lng":"Long Kick Returns",
                                    "kick_td":"Kick Return Touchdowns",
                                    "punt_att":"Punt Returns",
                                    "punt_yds":"Punt Return Yds.",
                                    "punt_avg":"Yds. Per Punt Return",
                                    "punt_lng":"Long Punt Return",
                                    "punt_td":"Punt Return Touchdown",
                                    "punt_fc":"Punt Return Fair Catches",
                                    "turn_diff":"Turnover Ratio",
                                    "take_int":"Takeaway Interceptions",
                                    "take_fum":"Fumbles Recovered",
                                    "take_total":"Total Takeaways",
                                    "give_int":"Giveaway Interceptions",
                                    "give_fum":"Fumbles Lost",
                                    "give_total":"Total Giveaways",
                                    "record": "Record",
                                    "rat":"Massey Overall",
                                    "pwr":"Massey Power",
                                    "off":"Massey Offense",
                                    "def":"Massey Defense",
                                    "hfa":"Massey Homefield",
                                    "sos":"Massey Schedule"
})

X = data.drop(["Year", "Playoff"], axis = 1)
y = data["Playoff"]

treeFinal = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=0.015)
treeFinal = treeFinal.fit(X, y)

st.title("Predict your team's probability of making the playoff!")

st.write("Use the drop down to enter in your teams stats. Working on file upload but running into auth issues!")

with st.expander("Click here to manually input a team stats."):
    col3,col4 = st.columns(2)
    with st.form(key = "manual_input"):
        Team = col3.text_input("Team Name")
        offTotYds = col4.number_input(label = "Offensive Total Yds. Per Game")
        offPassYds = col3.number_input(label = "Offensive Passing Yds. Per Game")
        offRushYds = col4.number_input(label = "Offensive Rushing Yds. Per Game")
        offPts = col3.number_input(label = "Offensive Points Per Game")
        defTotYds = col4.number_input(label = "Defensive Total Yds. Per Game")
        defPassYds = col3.number_input(label = "Defensive Passing Yds. Per Game")
        defRushYds = col4.number_input(label = "Defensive Rushing Yds. Per Game")
        defPts = col3.number_input(label = "Defensive Points Per Game")
        kickAtt = col4.number_input(label = "Kick Returns")
        kickYds = col3.number_input(label = "Kick Return Yds.")
        kickAvg = col4.number_input(label = "Yds. Per Kick Return")
        kickLng = col3.number_input(label = "Long Kick Return")
        kickTd = col4.number_input(label = "Kick Return Touchdowns")
        puntAtt = col3.number_input(label = "Punt Returns")
        puntYds = col4.number_input(label = "Punt Return Yds.")
        puntAvg = col3.number_input(label = "Yds. Per Punt Return")
        puntLng = col4.number_input(label = "Long Punt Return")
        puntTd = col3.number_input(label = "Punt Return Touchdown")
        puntFc = col4.number_input(label = "Punt Return Fair Catches")
        turnDiff = col3.number_input(label = "Turnover Ratio")
        takeInt = col4.number_input(label = "Takeaway Interceptions")
        takeFum = col3.number_input(label = "Fumbles Recovered")
        takeTot = col4.number_input(label = "Total Takeaways")
        giveInt = col3.number_input(label = "Giveaway Interceptions")
        giveFum = col4.number_input(label = "Fumbles Lost")
        giveTot = col3.number_input(label = "Total Giveaways")
        record = col4.number_input(label = "Record")
        rat = col3.number_input(label = "Massey Overall Rating")
        pwr = col4.number_input(label = "Massey Power Rating")
        off = col3.number_input(label = "Massey Offensive Rating")
        deff = col4.number_input(label = "Massey Defensive Rating")
        hfa = col3.number_input(label = "Massey Homefield Advantage Rating")
        sos = col4.number_input(label = "Massey Strength of Schedule Rating")
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            dataList = {
                "Team":Team,
                "Offensive Total Yds. Per Game":[offTotYds],
                "Offensive Passing Yds. Per Game":[offPassYds], 
                "Offensive Rushing Yds. Per Game":[offRushYds], 
                "Offensive Points Per Game":[offPts], 
                "Defensive Total Yds. Per Game":[defTotYds], 
                "Defensive Passing Yards Per Game":[defPassYds], 
                "Defensive Rushing Yards Per Game":[defRushYds],
                "Defensive Points Per Game":[defPts],
                "Kick Returns":[kickAtt],
                "Kick Return Yds.":[kickYds],
                "Yds. Per Kick Return":[kickAvg],
                "Long Kick Returns":[kickLng],
                "Kick Return Touchdowns":[kickTd],
                "Punt Returns":[puntAtt],
                "Punt Return Yds.":[puntYds],
                "Yds. Per Punt Return":[puntAvg],
                "Long Punt Return":[puntLng],
                "Punt Return Touchdown":[puntTd],
                "Punt Return Fair Catches":[puntFc],
                "Turnover Ratio":[turnDiff],
                "Takeaway Interceptions":[takeInt],
                "Fumbles Recovered":[takeFum],
                "Total Takeaways":[takeTot],
                "Giveaway Interceptions":[giveInt],
                "Fumbles Lost":[giveFum],
                "Total Giveaways":[giveTot],
                "Record":[record],
                "Massey Overall":[rat],
                "Massey Power":[pwr],
                "Massey Offense":[off],
                "Massey Defense":[deff],
                "Massey Homefield":[hfa],
                "Massey Schedule":[sos]
            }
            data = pd.DataFrame(dataList)
            
            if treeFinal.predict(data.drop("Team", axis = 1)).item() == 1:
                st.subheader(f"{Team} Predicted Playoff Result:")
                st.title(":blue[Yes]")
            else:
                st.subheader(f"{Team} Predicted Playoff Result:")
                st.title(":red[No]")
                             