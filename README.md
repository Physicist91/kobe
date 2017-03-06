# Kobe Bryant Shot Selection

This project analysed the Kaggle playground challenge [Kobe Bryant's shot selection](https://www.kaggle.com/c/kobe-bryant-shot-selection). The data contained the location and circumstances of every field goal attempted by Kobe Bryant during his 20-year career. The task was to predict whether the basket went in (shot_made_flag). Kaggle had removed 5000 of the shot_made_flags (represented as missing values in the csv file). These were the test set shots for which one must submit a prediction.

Please find "main.pdf" for the summary report of this analysis.

```
.
+-- _main.Rmd 
+-- _data <- The dataset provided by Kaggle for the challenge
+-- _code <- Jupyter notebooks and Python/R scripts 
+-- _submission <- submission files to the competition
+-- main.pdf <- The main PDF report produced by R Markdown
```

## ML Models

Various models were run for this project:

1. XGBOOST in Python
2. XGBOOST in R
2. TensorFLOW DNN in Python
3. sklearn (random forest, logistic regression, linear discriminant analysis, K-NN, decision tree, naive bayes, extra tree, adaboost, and gbm) in Python

## Resources

The following tools/platforms were used for this analysis:

1. Digital Ocean
2. Kaggle's kernel
3. Google Cloud Platform's docker
4. Python and R (with Jupyter Notebooks, RStudio, and RMarkdown)
5. x86_64-apple-darwin13.4.0 (64-bit)

## References

I would like to credit the following Exploratory Data Analysis notebooks upon which these analyses were built:

* https://www.kaggle.com/khozzy/kobe-bryant-shot-selection/kobe-shots-show-me-your-best-model
* https://www.kaggle.com/selfishgene/kobe-bryant-shot-selection/psychology-of-a-professional-athlete
* https://www.kaggle.com/apapiu/kobe-bryant-shot-selection/exploring-kobe-s-shots