def loadData():
    import pandas as pd
    path = "../../nfl-playoff-predictor/data/nfl_data.xlsx"
    df = pd.read_excel(path)
    return df

