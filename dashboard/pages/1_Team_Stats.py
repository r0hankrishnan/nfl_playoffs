#Libs
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

#Paths 
path = "./data/modelling.csv"
path2 = "./data/nfl_data.xlsx"

#Load data
@st.cache_data
def load_data():
    df = pd.read_csv(path)
    teams = pd.read_excel(path2)
    df = df.drop("Unnamed: 0", axis = 1)
    df["Team"] = teams["Team"]
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('Team')))
    cols.insert(1, cols.pop(cols.index('Playoff')))
    cols.insert(2, cols.pop(cols.index('Year')))
    df = df.loc[:, cols]
    return df

#Load data, copy df, convert Year to str
data = load_data()
displayData = data.copy()
displayData["Year"] = displayData["Year"].astype(str)

#Sidebar
teamChoice = displayData["Team"].unique()
yearChoice = displayData["Year"].unique()
with st.sidebar:
    st.header("Pick a team and year.")
    teamFilter = st.selectbox(label = "Pick a team", options = teamChoice)
    yearFilter = st.selectbox(label = "Pick a year", options = yearChoice)
    
#Title
st.title("Explore a specific team's stats!")

#Get win record as a number 
winRecord = displayData[(displayData["Team"] == teamFilter)&
                                                (displayData["Year"] == yearFilter)]["record"].values.astype(float).item()
#Display win record   
st.subheader(f"{teamFilter} {yearFilter} Win Record: {winRecord * 100} %")
  
#Make 3 columns to display Massey Ratings
col1,col2, col3 = st.columns(3)

col1.metric(label = "Massey Overall Rating", value = displayData[(displayData["Team"] == teamFilter)&
                                                (displayData["Year"] == yearFilter)]["rat"].values)
col1.metric(label = "Massey Power Rating", value = displayData[(displayData["Team"] == teamFilter)&
                                                (displayData["Year"] == yearFilter)]["pwr"].values)
col2.metric(label = "Massey Offensive Rating", value = displayData[(displayData["Team"] == teamFilter)&
                                                (displayData["Year"] == yearFilter)]["off"].values)
col2.metric(label = "Massey Defensive Rating", value = displayData[(displayData["Team"] == teamFilter)&
                                                (displayData["Year"] == yearFilter)]["def"].values)
col3.metric(label = "Massey Home Field Advantage Rating", value = displayData[(displayData["Team"] == teamFilter)&
                                                (displayData["Year"] == yearFilter)]["hfa"].values)
col3.metric(label = "Massey Strength of Schedule Rating", value = displayData[(displayData["Team"] == teamFilter)&
                                                (displayData["Year"] == yearFilter)]["sos"].values)

#Bar plot of per game stats
gameData = displayData[["Team", "Year", "off_total_yds_g", "off_pass_yds_g", "off_rush_yds_g", 
                "off_pts_g", "def_total_yds_g", "def_pass_yds_g", "def_rush_yds_g",
                "def_pts_g"]]

yearList = [yearFilter, str(int(yearFilter)-1)]

gameFiltered = gameData[(gameData["Team"] == teamFilter)&
                        (gameData["Year"].isin(yearList))]
                        

gameFiltered = gameFiltered.rename(columns = {"off_total_yds_g":"Offensive Total Yds. Per Game",
                                    "off_pass_yds_g":"Offensive Passing Yds. Per Game", 
                                    "off_rush_yds_g":"Offensive Rushing Yds. Per Game", 
                                    "off_pts_g":"Offensive Points Per Game", 
                                    "def_total_yds_g":"Defensive Total Yds. Per Game", 
                                    "def_pass_yds_g":"Defensive Passing Yards Per Game", 
                                    "def_rush_yds_g": "Defensive Rushing Yards Per Game",
                                    "def_pts_g":"Defensive Points Per Game"
    
})

gameMelted = pd.melt(gameFiltered, id_vars = ["Team", "Year"],
                     value_vars = ["Offensive Total Yds. Per Game",
          "Offensive Passing Yds. Per Game",
          "Offensive Rushing Yds. Per Game",
          "Offensive Points Per Game",
          "Defensive Total Yds. Per Game",
          "Defensive Passing Yards Per Game",
          "Defensive Rushing Yards Per Game",
          "Defensive Points Per Game"])

fig = px.bar(gameMelted, x = "value", y = "variable", color="Year", barmode = "group",
             labels = {
                 "value":"",
                 "variable":""
             },
             text_auto = True)

st.plotly_chart(fig)

#Look at other stats over time
statFiltered = displayData[(displayData["Team"]==teamFilter)]

statFiltered = statFiltered.rename(columns = {"off_total_yds_g":"Offensive Total Yds. Per Game",
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

varChoices = statFiltered.drop(["Playoff","Year","Team",],axis=1).columns
with st.sidebar:
    st.divider()
    st.header("Look at any metric from 2014-2022.")
    varFilter = st.selectbox(label = "Pick a stat", options = varChoices)

fig2 = px.line(statFiltered[(gameData["Team"] == teamFilter)], x = "Year", y = varFilter)
st.plotly_chart(fig2)