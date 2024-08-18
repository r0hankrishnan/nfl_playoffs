#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

#Set paths for data
path = "./data/modelling.csv"
path2 = "./data/nfl_data.xlsx"

#--------------#

#Title
st.title('🏈 NFL Playoff Dashboard')
st.divider()
st.write("Welcome to my simple NFL playoff predictor dashboard! You can filter the table and vizualization using the fields in the sidebar.")

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
    st.header("🔎 Filters:")
    hideData = st.checkbox("Hide Data Table")
    hideBar = st.checkbox("Hide Massey Ratings Viz")
    hideScatter = st.checkbox("Hide ESPN Stats Viz")
    yearFilter = st.selectbox(label = "Pick a Year", options = yearList)
    playoffFilter = st.selectbox(label = "Playoffs (Yes/No)", options = ["All", "Yes", "No"])

#---------------#

#Stats Table

#Function to filter data based on sidebar values
def filterData():
          
    #If only year is "All"  
    if yearFilter == "All":
        if playoffFilter == "Yes":
            filteredData = displayData[(displayData["Playoff"] == 1)]
        elif playoffFilter == "All":
            filteredData = displayData
        else:
            filteredData = displayData[(displayData["Playoff"] == 0)]

    #If only team is "All"        
    else:
        if playoffFilter == "Yes":
            filteredData = displayData[(displayData["Year"] == yearFilter) &
                                       (displayData["Playoff"] == 1)]
        elif playoffFilter == "All":
            filteredData = displayData[(displayData["Year"] == yearFilter)]
        else:
            filteredData = displayData[(displayData["Year"] == yearFilter) &
                                       (displayData["Playoff"] == 0)]
    
    return filteredData

#Load filtered data
filteredTable = filterData()

#Display stats table
st.subheader(f"🧾 Stats (Year: {yearFilter} | Playoff: {playoffFilter})")
if not hideData:
    st.write(filteredTable)

#---------------#

#Massey Ratings
st.divider()
st.subheader(f"📊 Massey Ratings (Year: {yearFilter} | Playoff: {playoffFilter})")

#Function to create grouped data
def generateGraphData():
    newData = data.drop(["Team"], axis = 1).groupby(["Playoff", "Year"]).mean(numeric_only=True)
    return newData

graphData = generateGraphData()

#Turn data into normal df, melt by playoff and year, change 1/0 to yes/no
graphData = pd.DataFrame(graphData.to_records()).reset_index()
masseyMelted = pd.melt(graphData, id_vars= ["Playoff", "Year"], 
                                 value_vars=["rat", "pwr", "def", "hfa", "sos"])
masseyMelted["Playoff"] = masseyMelted["Playoff"].astype(str)
masseyMelted["Playoff"] = pd.Series(np.where(masseyMelted["Playoff"].values == '1', "Yes", "No"),
          masseyMelted.index)

#Create filtering logic 
def filterGraphData():
    if yearFilter == "All" and playoffFilter == "All":
        filteredDf = masseyMelted
        
    elif yearFilter != "All" and playoffFilter == "All":
        filteredDf = masseyMelted[(masseyMelted["Year"] == int(yearFilter))]
        
    elif yearFilter == "All" and playoffFilter != "All":
        if playoffFilter == "Yes":
            filteredDf = masseyMelted[(masseyMelted["Playoff"] == playoffFilter)]
        else:
            filteredDf = masseyMelted[(masseyMelted["Playoff"] == playoffFilter)]
  
    else:
        if playoffFilter == "Yes":
            filteredDf = masseyMelted[(masseyMelted["Year"] == int(yearFilter)) &
                                      (masseyMelted["Playoff"] == playoffFilter)]
        else:
            filteredDf = masseyMelted[(masseyMelted["Year"] == int(yearFilter)) &
                                      (masseyMelted["Playoff"] == playoffFilter)]
    
    return filteredDf
            
#Generate filtered data
masseyFiltered = filterGraphData()

#Create side-by-side bar plot and display
fig = px.bar(masseyFiltered, x="value", y="variable", color="Playoff", barmode="group",
             labels={
                 "value":"Value",
                 "variable":"Massey Ratings"
             }
             )
fig.update_layout(legend_traceorder="reversed")

if not hideBar:
    st.plotly_chart(fig)

#----------#

#Scatter Matrix
st.divider()
st.subheader(f"📈 ESPN Per-Game Stats (Year: {yearFilter} | Playoff: {playoffFilter})")

#Create scatter matrix data
scatterData = data.drop(["Team", "rat", "pwr", "def", "hfa", "sos"], axis = 1)
scatterData["Playoff"] = scatterData["Playoff"].astype(str)
scatterData["Playoff"] = pd.Series(np.where(scatterData["Playoff"].values == '1', "Yes", "No"),
          scatterData.index)
scatterData = scatterData[["Year", "Playoff", "off_total_yds_g", "off_pass_yds_g", "off_rush_yds_g", 
                "off_pts_g", "def_total_yds_g", "def_pass_yds_g", "def_rush_yds_g",
                "def_pts_g"]]

#Create filtering logic
def filterScatterData():
    if yearFilter == "All" and playoffFilter == "All":
        filteredDf = scatterData
        
    elif yearFilter != "All" and playoffFilter == "All":
        filteredDf = scatterData[(scatterData["Year"] == int(yearFilter))]
        
    elif yearFilter == "All" and playoffFilter != "All":
        if playoffFilter == "Yes":
            filteredDf = scatterData[(scatterData["Playoff"] == playoffFilter)]
        else:
            filteredDf = scatterData[(scatterData["Playoff"] == playoffFilter)]
  
    else:
        if playoffFilter == "Yes":
            filteredDf = scatterData[(scatterData["Year"] == int(yearFilter)) &
                                      (scatterData["Playoff"] == playoffFilter)]
        else:
            filteredDf = scatterData[(scatterData["Year"] == int(yearFilter)) &
                                      (scatterData["Playoff"] == playoffFilter)]
    
    return filteredDf

#Load filtered data
scatterFilter = filterScatterData()

#Generate scatter plot, adjust labels & sizes, display
fig2 = px.scatter_matrix(scatterFilter,
    dimensions=["off_total_yds_g", "off_pass_yds_g", "off_rush_yds_g", 
                "off_pts_g", "def_total_yds_g", "def_pass_yds_g", "def_rush_yds_g",
                "def_pts_g"],
    color="Playoff",
    size_max = 1,
    labels = {
        "off_total_yds_g":"Offense Tot. Yds.",
        "off_pass_yds_g":"Offense Pass. Yds.",
        "off_rush_yds_g":"Offense Rush Yds.",
        "off_pts_g":"Offense Pts.",
        "def_total_yds_g":"Defense Tot. Yds.",
        "def_pass_yds_g":"Defense Pass Yds.",
        "def_rush_yds_g":"Defense Rush Yds.",
        "def_pts_g":"Defense Pts."
    })
fig2.update_layout(yaxis1 = {"title":{"font":{"size":5}}}, yaxis2 = {"title":{"font":{"size":5}}},
                   yaxis3 = {"title":{"font":{"size":5}}}, yaxis4 = {"title":{"font":{"size":5}}},
                   yaxis5 = {"title":{"font":{"size":5}}}, yaxis6 = {"title":{"font":{"size":5}}},
                   yaxis7 = {"title":{"font":{"size":5}}}, yaxis8 = {"title":{"font":{"size":5}}})

fig2.update_layout(xaxis1 = {"title":{"font":{"size":7}}}, xaxis2 = {"title":{"font":{"size":7}}},
                   xaxis3 = {"title":{"font":{"size":7}}}, xaxis4 = {"title":{"font":{"size":7}}},
                   xaxis5 = {"title":{"font":{"size":7}}}, xaxis6 = {"title":{"font":{"size":7}}},
                   xaxis7 = {"title":{"font":{"size":7}}}, xaxis8 = {"title":{"font":{"size":7}}})

fig2.update_traces(marker=dict(size=3, opacity = 0.6))
fig2.update_layout(autosize=False,
    width=900,
    height=650)

if not hideScatter:
    st.plotly_chart(fig2)



