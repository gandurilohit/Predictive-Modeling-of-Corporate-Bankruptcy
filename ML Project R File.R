# 1. Set working directory
setwd("C:/Users/Lohit/Downloads/Machine Learning Project")  # <-- Replace this path


library(tidyverse)
data <- read.csv("C:/Users/Lohit/Downloads/HWFAILED (1).csv")

# Remove rows with NA (optional depending on strategy)
data_clean <- na.omit(data)

# Convert distress indicator to factor (for classification)
data_clean$yd <- as.factor(data_clean$yd)


# LASSO REGRESSION
library(glmnet)

# Prepare data matrices
X <- model.matrix(yd ~ . -1, data = data_clean)  # Remove intercept
y <- data_clean$yd

# Lasso with cross-validation
cv_lasso <- cv.glmnet(X, y, family = "binomial", alpha = 1)
plot(cv_lasso)

# Best lambda
best_lambda <- cv_lasso$lambda.min

# Coefficients
coef(cv_lasso, s = "lambda.min")


# Ridge with cross-validation
cv_ridge <- cv.glmnet(X, y, family = "binomial", alpha = 0)
plot(cv_ridge)

# Best lambda
best_lambda_ridge <- cv_ridge$lambda.min

# Coefficients at best lambda
ridge_coefs <- coef(cv_ridge, s = best_lambda_ridge)
print(ridge_coefs)



# Load required packages
install.packages("xtable")  # if not installed
library(xtable)

# Fit models
logit_model <- glm(yd ~ ., data = data_clean, family = binomial(link = "logit"))
probit_model <- glm(yd ~ ., data = data_clean, family = binomial(link = "probit"))

# Extract summaries
logit_sum <- summary(logit_model)
probit_sum <- summary(probit_model)

# Create combined dataframe
logit_probit_df <- data.frame(
  Variable = rownames(logit_sum$coefficients),
  Logit_Coeff = round(logit_sum$coefficients[, 1], 3),
  Logit_SE = round(logit_sum$coefficients[, 2], 3),
  Logit_Pval = signif(logit_sum$coefficients[, 4], 3),
  Probit_Coeff = round(probit_sum$coefficients[, 1], 3),
  Probit_SE = round(probit_sum$coefficients[, 2], 3),
  Probit_Pval = signif(probit_sum$coefficients[, 4], 3)
)

# View table in console
print(logit_probit_df)

# Export to LaTeX table
xtable_obj <- xtable(logit_probit_df, caption = "Logit vs. Probit Regression Results")

# Save to .tex file
sink("logit_probit_table.tex")
cat("\\begin{table}[ht]\n\\centering\n")
print(xtable_obj, include.rownames = FALSE, sanitize.text.function = identity)
cat("\\end{table}\n")
sink()



# RANDOM FOREST
install.packages("randomForest")
library(randomForest)

rf_model <- randomForest(yd ~ ., data = data_clean, importance = TRUE)
print(rf_model)
varImpPlot(rf_model)


# XGBoost
install.packages("xgboost")
library(xgboost)

# Reconvert to numeric (xgboost requires matrix)
X_xgb <- as.matrix(data_clean[, -which(names(data_clean) == "yd")])
y_xgb <- as.numeric(as.character(data_clean$yd))

dtrain <- xgb.DMatrix(data = X_xgb, label = y_xgb)

params <- list(objective = "binary:logistic", eval_metric = "auc")
xgb_model <- xgb.cv(params = params, data = dtrain, nrounds = 100, nfold = 5, verbose = 0)
print(xgb_model)


# Model Evaluation: ROC, AUC, Accuracy
library(caret)
library(pROC)

set.seed(123)
train_idx <- createDataPartition(data_clean$yd, p = 0.8, list = FALSE)
train <- data_clean[train_idx, ]
test <- data_clean[-train_idx, ]

# Fit model (example: logistic)
log_model <- glm(yd ~ ., data = train, family = "binomial")
probs <- predict(log_model, newdata = test, type = "response")
pred <- ifelse(probs > 0.5, 1, 0)

# Accuracy
confusionMatrix(as.factor(pred), test$yd)

# AUC
roc_obj <- roc(test$yd, probs)
auc(roc_obj)
plot(roc_obj)


# Neural Networks
install.packages("keras")
install.packages("tensorflow")
install.packages("keras")
library(keras)
install_keras()

library(keras)
library(tensorflow)
library(dplyr)




# Preprocessing
X <- scale(as.matrix(data_clean[, -1])) # Standardize predictors
y <- as.numeric(as.character(data_clean$yd)) # Ensure numeric 0/1

# Train/test split
set.seed(123)
idx <- sample(1:nrow(X), 0.8 * nrow(X))
X_train <- X[idx, ]; y_train <- y[idx]
X_test  <- X[-idx, ]; y_test <- y[-idx]

# Build model
model <- keras_model_sequential() %>%
  layer_dense(units = 8, activation = 'relu', input_shape = ncol(X)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = 'accuracy'
)

# Train
history <- model %>% fit(
  X_train, y_train,
  epochs = 100, batch_size = 8,
  validation_split = 0.2,
  callbacks = list(callback_early_stopping(patience = 10))
)

# Evaluate
model %>% evaluate(X_test, y_test)


