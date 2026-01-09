Forecasting Time Series Using Prophet and Adjusting Models
Final Project Submission

Project Overview

1. Dataset Characteristics
Synthetic Time Series Data,
From [Start date], moving through to [End date], 
Data Points Thousands of Daily Observations,
Seasonal Patterns: [Which patterns you included], 
External Regressors List.
On average, values sit around the mean. A spread shows how much they differ from one another. Some fall at the lowest point recorded. Others reach the highest seen so far

3. Implementation Approach

2.1 Data Preparation: Describe your data generation/preprocessing
2.2 Cross-Validation Strategy
Rolled Origin Cross Validation
- Initial training size: [e.g., 365 days]
- Horizon: [e.g., 60 days]
- Step size: [e.g., 30 days]
Folds total? Eight times it bends.

2.3 Hyperparameter Tuning
Final selected parameters:
- `changepoint_prior_scale`: [value]
- `seasonality_prior_scale`: [value]
- `holidays_prior_scale`: [value]
- `seasonality_mode`: [value]

2.4 Model Calibration
Does it require calibration? Yes
- Calibration method: [Isotonic regression/Quantile regression/Other]
Looking at two levels of coverage: one hits 80 percent, the other reaches 95. Each offers a different view on how much is truly included

3. Results

3.1 Forecast Accuracy
- MAE: [value]
- RMSE: [value]
- MAPE: [value]%

3.2 Initial Prediction Interval Coverage
- 80% Interval: [value]% coverage
Most of the time, around 95 out of every 100 cases fall within this range - [value] percent captured
Calibration slips show up as an 80 percent mistake, sometimes jumping to 95 percent off. Numbers drift without warning, throwing results sideways

3.3 Adjusted Forecast Range Accuracy
Most of the time - about eight out of ten - the range includes the true value. That actual number sits right at [value] percent when checked against real data
Most of the time, around 95 out of every 100 cases fall within this range after adjustments were made. That means roughly [value] percent are captured here
Most saw progress - around eight out of ten people felt noticeably better. Nearly everyone else reported strong results too

4. Code Implementation

Put your full Python code right here - just copy and drop it in

5. Visualizations
Picture a line showing what was expected versus what actually happened. See how much of the data fits within set boundaries. Look at differences between predicted numbers and real outcomes. Notice where predictions missed the mark by checking leftover gaps. Compare spans across different groups to spot shifts

6. Discussion

6.1 Key Findings
- [What worked well]
- [What was challenging]
- [Insights from calibration]

6.3 Limits and What Could Come Next
If given extra hours, changes might show up in small choices. Instead of rushing, pauses would appear between tasks. Decisions could breathe before they land. Speed often hides what care reveals slowly. Time lets details speak louder than urgency ever does
