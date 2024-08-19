import streamlit as st
import pandas as pd
from sklearn import tree

st.set_page_config(
        page_title='Predictions',
        page_icon="🔍"                  
        )

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
        Year = st.text_input(label = "Year", value = "2023")
        Team = col3.text_input("Team Name", value = "Miami Dolphins")
        offTotYds = col4.number_input(label = "Offensive Total Yds. Per Game", value = 513.6)
        offPassYds = col3.number_input(label = "Offensive Passing Yds. Per Game", value = 327.8)
        offRushYds = col4.number_input(label = "Offensive Rushing Yds. Per Game", value = 185.8)
        offPts = col3.number_input(label = "Offensive Points Per Game", value = 36.2)
        defTotYds = col4.number_input(label = "Defensive Total Yds. Per Game", value = 353.2)
        defPassYds = col3.number_input(label = "Defensive Passing Yds. Per Game", value = 237.4)
        defRushYds = col4.number_input(label = "Defensive Rushing Yds. Per Game", value = 115.8)
        defPts = col3.number_input(label = "Defensive Points Per Game", value = 27)
        kickAtt = col4.number_input(label = "Kick Returns", value = 5)
        kickYds = col3.number_input(label = "Kick Return Yds.", value = 116)
        kickAvg = col4.number_input(label = "Yds. Per Kick Return", value = 23.2)
        kickLng = col3.number_input(label = "Long Kick Return", value = 27)
        kickTd = col4.number_input(label = "Kick Return Touchdowns", value = 0)
        puntAtt = col3.number_input(label = "Punt Returns", value = 6)
        puntYds = col4.number_input(label = "Punt Return Yds.", value = 57)
        puntAvg = col3.number_input(label = "Yds. Per Punt Return", value = 9.5)
        puntLng = col4.number_input(label = "Long Punt Return", value = 18)
        puntTd = col3.number_input(label = "Punt Return Touchdown", value = 0)
        puntFc = col4.number_input(label = "Punt Return Fair Catches", value = 9)
        turnDiff = col3.number_input(label = "Turnover Ratio", value = -3)
        takeInt = col4.number_input(label = "Takeaway Interceptions", value = 2)
        takeFum = col3.number_input(label = "Fumbles Recovered", value = 3)
        takeTot = col4.number_input(label = "Total Takeaways", value = 5)
        giveInt = col3.number_input(label = "Giveaway Interceptions", value = 5)
        giveFum = col4.number_input(label = "Fumbles Lost", value = 3)
        giveTot = col3.number_input(label = "Total Giveaways", value = 8)
        record = col4.number_input(label = "Record", value = 0.80)
        rat = col3.number_input(label = "Massey Overall Rating", value = 8.84)
        pwr = col4.number_input(label = "Massey Power Rating", value = 3)
        off = col3.number_input(label = "Massey Offensive Rating", value = 27.01)
        deff = col4.number_input(label = "Massey Defensive Rating", value = -1.09)
        hfa = col3.number_input(label = "Massey Homefield Advantage Rating", value = 1.37)
        sos = col4.number_input(label = "Massey Strength of Schedule Rating", value = -0.56)
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

if submitted:        
    if treeFinal.predict(data.drop("Team", axis = 1)).item() == 1:
        with st.container(border = True):
            st.subheader(f"{Team} {Year} Predicted Playoff Result:")
            st.title(":blue[Yes]")
    else:
        with st.container(border = True):
            st.subheader(f"{Team} {Year} Predicted Playoff Result:")
            st.title(":red[No]")
                             