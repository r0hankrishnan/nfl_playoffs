import marimo

__generated_with = "0.7.8"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        r"""
        # Introduction

        This project will use a self-created excel dataset with data about which NFL teams made the playoffs from 2014 to 2022. The goal is to understand the data, clean it as needed, and develop several statistical models to try and predict which team will make the playoffs.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # EDA

        ## Import Modules
        """
    )
    return


@app.cell
def __():
    #Import marimo + eda modules

    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import openpyxl
    return mo, np, openpyxl, pd, plt, sns


@app.cell
def __(mo):
    mo.md(r"## Load Data & Basic Exploration")
    return


@app.cell
def __(pd):
    #Load data
    path = "data/hw2_nfl_final_data.xlsx"
    df = pd.read_excel(path)
    return df, path


@app.cell
def __(mo):
    mo.md(r"Let's take a quick look at our data set.")
    return


@app.cell
def __(df):
    df
    return


@app.cell
def __(mo):
    mo.md(r"Let's look at the dimensions, descriptive statistics, and name of the data set and its columns.")
    return


@app.cell
def __(df):
    print(f"Df size: {df.size} \nDf shape: {df.shape}\nDf ndim: {df.ndim}")
    return


@app.cell
def __(df):
    df.describe()
    return


@app.cell
def __(df):
    for col in df.columns:
        print(f"Index {df.columns.get_loc(col)}: {col}")
    return col,


@app.cell
def __(mo):
    mo.md(r"Let's only consider per game statistics for each team.")
    return


@app.cell
def __(mo):
    mo.md(r"For the purposes of our model, we only want to consider a team's **per game** performance. So, let's filter out all the columns corresponding to total values.")
    return


@app.cell
def __(df):
    dropCols = list(["Team","off_gp","off_total_yds","off_pass_yds","off_rush_yds","off_pts","def_total_yds"
                 ,"def_pass_yds","def_rush_yds","def_pts"])
    df1 = df.copy()
    df1 = df1.drop(dropCols, axis = 1)
    df1
    return df1, dropCols


@app.cell
def __(df1, mo):
    mo.md(
        f"Our new df has the following rows and columns: {df1.shape}."
    )
    return


@app.cell
def __(mo):
    mo.md(r"It looks like our df also has some columns with negative numbers in them. Let's adjust those columns up so their minimum is 0. ")
    return


@app.cell
def __(df1, mo):
    df1_negCols = []
    for i in df1.columns:
        if df1[i].min() < 0:
            df1_negCols.append(i)

    mo.md(
        f"The following columns have negative values {df1_negCols}."
    )


    return df1_negCols, i


@app.cell
def __(mo):
    mo.md(r"Now let's create a new data frame with those columns adjusted upwards.")
    return


@app.cell
def __(df1, df1_negCols):
    df2 = df1.copy()
    for a in df1_negCols:
        df2[a] = df2[a] - df2[a].min()

    df2[df1_negCols].describe()
    return a, df2


@app.cell
def __(mo):
    mo.md(r"Let's take a look at how each column correlates to our *Playoff* variable.")
    return


@app.cell
def __(df2, plt, sns):
    plt.title("Correlations")
    sns.heatmap(df2.corr(), cmap="coolwarm")
    return


@app.cell
def __(mo):
    mo.md(r"Seems like there are several variables that appear to be fairly highlight correlated! Let's list all the variables with an absolute correlation of *at least* 0.50 below.")
    return


@app.cell
def __(df2):
    correlations = df2.corr()
    correlations = correlations.loc[:, "Playoff"]
    correlations = correlations.reset_index().rename(columns = {"index":"variable"})
    correlations = correlations[(correlations["Playoff"] >0.50) | (correlations["Playoff"] < -0.50)]
    correlations.sort_values(by = "Playoff")
    return correlations,


if __name__ == "__main__":
    app.run()
