import marimo

__generated_with = "0.7.0"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Basic Data EDA

        ### Install pandas and openpyxl & import pandas and os
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    # pip install pandas
    # pip install openpyxl
    return


@app.cell
def __():
    import os
    import pandas as pd
    import openpyxl
    return openpyxl, os, pd


@app.cell
def __(pd):
    path = "/Users/rohankrishnan/Documents/GitHub/nfl-playoff-predictor/data/hw2_nfl_final_data.xlsx"
    nfl = pd.read_excel(path)
    nfl.head()
    return nfl, path


@app.cell
def __(mo):
    mo.md(r"### Move response to front of dataset")
    return


@app.cell
def __(nfl):
    cols = list(nfl.columns.values)

    nfl_1 = nfl[[ 'Playoff',
     'Team',
     'off_gp',
     'off_total_yds',
     'off_total_yds_g',
     'off_pass_yds',
     'off_pass_yds_g',
     'off_rush_yds',
     'off_rush_yds_g',
     'off_pts',
     'off_pts_g',
     'def_total_yds',
     'def_total_yds_g',
     'def_pass_yds',
     'def_pass_yds_g',
     'def_rush_yds',
     'def_rush_yds_g',
     'def_pts',
     'def_pts_g',
     'kick_att',
     'kick_yds',
     'kick_avg',
     'kick_lng',
     'kick_td',
     'punt_att',
     'punt_yds',
     'punt_avg',
     'punt_lng',
     'punt_td',
     'punt_fc',
     'turn_diff',
     'take_int',
     'take_fum',
     'take_total',
     'give_int',
     'give_fum',
     'give_total',
     'record',
     'rat',
     'pwr',
     'off',
     'def',
     'hfa',
     'sos',
     'Year',]]
    return cols, nfl_1


@app.cell
def __(mo):
    mo.md(r"### Get basic distributions of all variables and check basic characteristics")
    return


@app.cell
def __(nfl_1):
    nfl_1.describe()
    return


@app.cell
def __(nfl_1):
    #Make sure there are no NA values
    nfl_1.isna().sum()
    return


@app.cell
def __(nfl_1):
    #Check which variables are continuous and which are discrete
    nfl_1.nunique()
    return


@app.cell
def __(nfl_1):
    #Slightly more No than Yes-- makes sense since only 12 (and later 14) teams make it
    nfl_1["Playoff"].value_counts()
    return


@app.cell
def __(nfl_1):
    #Conditional means - some differences
    cond_means = nfl_1.groupby("Playoff").mean(numeric_only=True)
    cond_means
    return cond_means,


@app.cell
def __(mo):
    mo.md(r"### Install analysis packages")
    return


@app.cell
def __():
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns


@app.cell
def __(mo):
    mo.md(r"### Visualizations")
    return


@app.cell
def __(nfl):
    corr_mat = nfl.drop("Team", axis  = 1).corr()
    return corr_mat,


@app.cell
def __(corr_mat, plt, sns):
    #Correlation Matrix
    plt.figure(figsize =(25, 20))
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    return


@app.cell
def __(nfl_1, plt, sns):
    #Scatter plot
    plt.figure(figsize = (25,20))
    sns.scatterplot(data = nfl_1,
                   x = "Playoff",
                   y = "off_total_yds")
    return


if __name__ == "__main__":
    app.run()
