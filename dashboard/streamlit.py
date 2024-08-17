#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import os

#Set paths for data
path = "./data/modelling.csv"
path2 = "./data/nfl_data.xlsx"
#--------------#

#Title
st.title('🏈 NFL Playoff Dashboard')

#Save loaded data
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

#Create selection options for table filters
teamList = displayData["Team"].unique()
teamList = np.insert(teamList, 0, "All")

yearList = displayData["Year"].unique()
yearList = np.insert(yearList, 0, "All")

#--------------#

#Sidebar
with st.sidebar:
    st.header("🔎 Table Filters:")
    hideData = st.checkbox("Hide Data Table")
    teamFilter = st.selectbox(label = "Pick a Team", options = teamList)
    yearFilter = st.selectbox(label = "Pick a Year", options = yearList)
    playoffFilter = st.selectbox(label = "Playoffs (Yes/No)", options = ["All", "Yes", "No"])
    st.divider()
    st.header("📈 Viz Filters:")
    
#Function to filter data based on sidebar values
def filterData():
    #If both team and year are "All"
    if teamFilter == "All" and yearFilter == "All":
        if playoffFilter == "Yes":
            filteredData = displayData[(displayData["Playoff"] == 1)]
        elif playoffFilter == "All":
            filteredData = displayData
        else:
            filteredData = displayData[(displayData["Playoff"]== 0)]
            
    #If only year is "All"  
    elif teamFilter != "All" and yearFilter == "All":
        if playoffFilter == "Yes":
            filteredData = displayData[(displayData["Team"] == teamFilter) &
                                       (displayData["Playoff"] == 1)]
        elif playoffFilter == "All":
            filteredData = displayData[(displayData["Team"] == teamFilter)]
        else:
            filteredData = displayData[(displayData["Team"] == teamFilter) &
                                       (displayData["Playoff"] == 0)]

    #If only team is "All"        
    elif teamFilter == "All" and yearFilter != "All":
        if playoffFilter == "Yes":
            filteredData = displayData[(displayData["Year"] == yearFilter) &
                                       (displayData["Playoff"] == 1)]
        elif playoffFilter == "All":
            filteredData = displayData[(displayData["Year"] == yearFilter)]
        else:
            filteredData = displayData[(displayData["Year"] == yearFilter) &
                                       (displayData["Playoff"] == 0)]
    
    #If specific team and year are selected   
    else: 
        if playoffFilter == "Yes":
            filteredData = displayData[(displayData["Team"] == teamFilter) &
                                        (displayData["Year"] == yearFilter) &
                                        (displayData["Playoff"] == 1)]
        elif playoffFilter == "All":
            filteredData = displayData[(displayData["Team"] == teamFilter)&
                                       (displayData["Year"] == yearFilter)]
        else:
            filteredData = displayData[(displayData["Team"] == teamFilter) &
                                        (displayData["Year"] == yearFilter) &
                                        (displayData["Playoff"] == 0)]
    
    return filteredData

#Load filtered data
filteredData = filterData()

#--------------#

#Display stats table
st.subheader(f"Stats ({teamFilter}, {yearFilter})")
if not hideData:
    st.write(filteredData)

