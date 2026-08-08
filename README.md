# NBA Analytics & Game Prediction Platform

A data analytics and machine learning project that combines SQL, exploratory analysis, and predictive modeling to study NBA player performance.

I built this project for Rutgers' CS210 course, but treated it as more than just a database assignment. The goal was to build a complete workflow starting with relational data, moving through SQL analysis and visualization, and ending with a machine learning model that could make predictions about future player performance.

## Project Overview

The project has three main parts:

1. Relational database and SQL analysis
2. Exploratory data analysis
3. Machine learning

The dataset contains 2,000+ season-level player records with statistics such as points, rebounds, assists, shooting efficiency, games played, draft information, and player demographics.

## Database

The data is stored in SQLite using three normalized tables:

```text
players
   |
   +---- player_stats
   |
teams
```

The schema separates player information, team information, and season-level statistics.

The database also includes indexes for common queries, including player-by-season lookups.

SQL analysis uses:

* JOINs
* GROUP BY
* HAVING
* CTEs
* Window functions
* `LAG()`
* Subqueries

One example uses `LAG()` to compare a player's scoring between consecutive seasons and identify year-over-year changes.

## Exploratory Analysis

The project includes visualizations examining questions such as:

* How does scoring change throughout a player's career?
* How common is the sophomore jump?
* How does scoring vary by age and position?
* Does draft position relate to long-term scoring?
* How consistent is scoring from one season to the next?
* How consistent is player availability?

These visualizations were used to explore the data before building the predictive models.

## Machine Learning

The ML portion uses a temporal train, validation, and test split.

This was important because randomly splitting NBA seasons would allow information from future seasons to enter the training data.

The model was trained on earlier seasons, validated on 2019-2020, and tested on 2021-2022.

### Classification

A Random Forest classifier predicts whether a player will become a high scorer in the following season.

The target is defined as the top 15th percentile of scoring.

Features include:

* Previous-season statistics
* Career averages
* Age
* Seasons in the league
* Years since draft
* Draft tier
* Height category
* Efficiency metrics

The classifier achieved 73% test accuracy.

### Regression

A Random Forest regressor predicts next-season fantasy points using a custom scoring formula based on:

* Points per game
* Rebounds per game
* Assists per game

The model was compared against a simple previous-season baseline to determine whether the machine learning model actually provided useful predictive improvement.

## Why Temporal Splitting?

For this type of problem, randomly shuffling the data would make the evaluation look better than it should.

If the model trains on 2022 data and then predicts a 2019 season, it is effectively using information from the future.

Using a temporal split produces a more realistic estimate of how the model would perform if it were actually making predictions before the next season.

The lower accuracy that results is useful because it gives a more honest picture of model performance.

## Tech Stack

* Python
* SQLite
* pandas
* scikit-learn
* Matplotlib
* seaborn

## Key Results

* 2,000+ player-season records
* 20+ engineered features
* 6 exploratory visualizations
* 2 Random Forest models
* 73% classification accuracy
* Temporal train/validation/test split
* Complex SQL analysis using CTEs and window functions

## What I Learned

The biggest takeaway was that building the model was only one part of the problem.

Data organization and evaluation methodology had just as much impact on the final result. In particular, using temporal validation made the model's performance much more realistic than a standard random train/test split.

The project also gave me a good excuse to combine three things I enjoy working with: SQL, data analysis, and machine learning.
