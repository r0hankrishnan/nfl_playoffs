# NFL Playoff Prediction Project

This project explores data-driven methods to predict which NFL teams are likely to make the playoffs in a given season. The full pipeline includes data collection, cleaning, exploratory analysis, predictive modeling, and the deployment of an interactive dashboard built with Streamlit.

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation and Background](#motivation-and-background)
- [Data Collection](#data-collection)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling Approach](#modeling-approach)
- [Dashboard](#dashboard)
- [Technologies Used](#technologies-used)
- [How to Reproduce](#how-to-reproduce)
- [Results and Discussion](#results-and-discussion)
- [Future Improvements](#future-improvements)

## Project Overview

In this project, I take on the role of a data analyst at a hypothetical sports betting company. The task is to develop a model that classifies NFL teams as either likely or unlikely to make the playoffs in a given season. Accurate predictions can help the company offer odds that are more profitable by identifying playoff contenders early.

## Motivation and Background

The American Gaming Association reported that sportsbook revenue in the U.S. reached $7.5 billion in 2022. Industry forecasts estimate a market size exceeding $40 billion by 2030. To remain competitive, sports betting firms require robust predictive tools that complement human judgment.

This project seeks to build a predictive model using NFL performance data and expert team ratings to forecast playoff outcomes.

## Data Collection

The dataset was manually curated from two main sources:

### ESPN NFL Statistics (2014–2022)
- Offensive stats (points per game, yards)
- Defensive stats (points allowed, takeaways)
- Turnover and special teams metrics
- Playoff qualification labels

### Massey Ratings
- Power ratings
- Offensive and defensive performance scores
- Win-loss record estimates
- Bayesian and regression-based expert evaluation

The data was gathered into Excel spreadsheets by year and merged using `INDEX(MATCH())` across five sheets into a single dataset. A separate dataset was constructed for the 2023 season to serve as the prediction target.

Final dataset size:  
- 32 teams × 9 years = 288 data points  
- 33 features after cleaning

## Exploratory Data Analysis

- **Distribution analysis:** Most features followed approximately normal or t-distributions.
- **Correlation analysis:** Key features correlated with playoff qualification included:
  - Defensive points per game
  - Turnover ratio
  - Massey power and overall ratings
  - Team record

- **Standardized comparisons:** Playoff and non-playoff teams were compared across seasons using standardized feature means. These comparisons confirmed the importance of the features listed above.

- **Collinearity:** Pairplots revealed some multicollinearity. Since tree-based models inherently handle feature selection, no additional dimensionality reduction was performed.

- **Team comparison plots:** Year-over-year bar plots for individual team performance were developed and later incorporated into the dashboard.

**You can view examples of exploratory plots in the './assets' directory.**

## Modeling Approach

Five models were tested:

| Model                               | Accuracy    |
|------------------------------------|-------------|
| Logistic Regression                | 59%         |
| Decision Tree                      | 84%         |
| Pruned Decision Tree (CCP)         | **84.38%**  |
| Random Forest                      | 70%         |
| Random Forest (Grid Search CV)     | 78%         |

Key considerations:
- Logistic regression struggled due to non-normal feature distributions.
- K-Nearest Neighbors was excluded due to the high feature dimensionality.
- Tree-based models performed best, with the pruned decision tree achieving the highest accuracy.
- Random forests exhibited overfitting, likely due to the small and relatively stable dataset.

The final selected model was the pruned decision tree, balancing performance with interpretability and speed.

## Dashboard

A primary learning goal of this project was to implement an interactive dashboard using [Streamlit](https://streamlit.io). The dashboard allows users to:

- Explore team statistics by year
- Visualize team performance changes over time
- Generate playoff predictions for the 2023 season using the final model

You can tour a demo of the app by clicking the link in the repository description (or [right here](https://r0hankrishnan-nfl.streamlit.app/)) and booting up the streamlit instance as prompted. You can also clone this repository and run the dashboard locally.

## Technologies Used

- Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)
- Streamlit
- Excel (for data preparation)
- Jupyter Notebook

## How to Reproduce

To run the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/nfl-playoff-predictor.git
   cd nfl-playoff-predictor
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the dashboard:

   ```bash
   streamlit run dashboard.py
   ```

## Results and Discussion

* Tree-based models proved most effective, with the decision tree outperforming more complex models in this case.
* Massey Ratings were highly predictive of playoff outcomes, validating the inclusion of expert-derived variables.
* A decision tree model provided explainability and speed, ideal for decision-making in operational settings.

## Future Improvements

* Automate data ingestion via web scraping or API pipelines
* Evaluate ensemble models (e.g., XGBoost, LightGBM)
* Integrate probability-based classification rather than binary outcomes
* Extend the dashboard to simulate hypothetical team improvements and their impact on playoff likelihood
