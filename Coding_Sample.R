#import packages and file
library(glmnet)
library(ggplot2)
library(reshape2)
library(caret)
library(tidyverse)
library(regclass)
library(dplyr)
library(pracma)
library(gsignal)
library(stats)
library(betareg)
library(lmtest)
library(sandwich)
tdf <- read.csv("/Users/martrinmunoz/Desktop/EconPredoc/Writing Samples/TdF/tdf_cleaned.csv")
# check distributions
summary(tdf)
hist(tdf$ped_tot)
# removing gen_ad_test because it is a constant
vars_to_remove <- c('gen_ad_test')
tdf <- tdf[, !(colnames(tdf) %in% vars_to_remove)]
# start by creating correlation matrix
heatmap <- function(df, vars) {
  cormat <- round(cor(na.omit(df)),2)
  # set up hierarchical clustering
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <- cormat[hc$order, hc$order]
  # remove redundant information
  cormat[lower.tri(cormat)] <- NA
  # melt cormat
  melted_cormat <- melt(cormat, na.rm = TRUE)
  # create matrix
  heat_plot <- ggplot(melted_cormat, aes(Var2, Var1, fill = value)) +
    geom_tile(color = "white") +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                         midpoint = 0, limit = c(-1,1), space = "Lab",
                         name = "Pearson\nCorrelation") +
    theme_minimal() + 
    theme(axis.text.x = element_text(angle = 90, vjust = 0, size = 7, hjust = 0),
    axis.text.y = element_text(size = 7)) +
    coord_fixed()
    colnames(melted_cormat) <- c('Var1', 'Var2', 'correlation')
    melted_cormat <- melted_cormat[order(melted_cormat$Var1),]
    return(list(heat_plot = heat_plot, cormat = melted_cormat))
}

heatmap(df = tdf, vars = colnames(tdf))$heat_plot
# list variables with correlations higher than |0.7|
tdf_cor <- heatmap(df = tdf, vars = colnames(tdf))$cormat
as.name('Varibles with correlations higher than |0.7|')
for(i in 1:nrow(tdf_cor)) {
  if (abs(tdf_cor[i, 'correlation']) > 0.7 & 
          tdf_cor[i, 'Var1'] != tdf_cor[i, 'Var2']) {
    print(as.name(paste(' ', tdf_cor[i, 'Var1'], 'and', tdf_cor[i, 'Var2'],
    'have correlation', tdf_cor[i, 'correlation'])))
  }
}
# also look at VIF
model1 <- lm(ped_tot~., data = tdf)
VIF(model1)
# let's remove avg_speed, num_entrants, first_prize_money, tot_time_winner and year
model1_remove <- c('avg_speed', 'num_entrants', 'first_prize_money', 
                   'tot_time_winner', 'year')
tdf <- tdf[, !(colnames(tdf) %in% model1_remove)]
# let's regress model and check VIF again
model2 <- lm(ped_tot~., data = tdf)
VIF(model2)
# VIF's look good now, but let's check correlations again
heatmap(df = tdf, vars = colnames(tdf))$heat_plot
# list variables with correlations higher than |0.7|
tdf_cor <- heatmap(df = tdf, vars = colnames(tdf))$cormat
for(i in 1:nrow(tdf_cor)) {
  if (abs(tdf_cor[i, 'correlation']) > 0.7 & 
      tdf_cor[i, 'Var1'] != tdf_cor[i, 'Var2']) {
    print(as.name(paste(' ', tdf_cor[i, 'Var1'], 'and', tdf_cor[i, 'Var2'],
                        'have correlation', tdf_cor[i, 'correlation'])))
  }
}
# we have 0.78 cor between epo_test and ooct and 0.89 between bio_passport and ooct
summary(model2)
#let's do model diagnostics for the manual model
par(mfrow = c(2, 2))
plot(model2)
#it appears that there is heteroskedasticity
# try to avoid heteroskedasticity by respecifying model2 with log
model3 <- lm(log(ped_tot)~., data = tdf)
summary(model3)
# model diagnostics again
par(mfrow = c(2, 2))
plot(model3)
#that didn't work, let's just get robust standard errors
coeftest(model2, vcov = vcovHC(model2, type = "HC1"))
#looks similar

#Now we try automatic feature selection via a LASSO loop
tdf <- read.csv("/Users/martrinmunoz/Desktop/EconPredoc/Writing Samples/TdF/tdf_cleaned.csv")
X_train <- model.matrix(ped_tot~., data = tdf)[,-1]
Y_train <- tdf$ped_tot
MSEs <- NULL
for (i in 1:100){
  cv <- cv.glmnet(x = X_train, y = Y_train, alpha=1, nfolds=5, standardize = TRUE)  
  MSEs <- cbind(MSEs, cv$cvm)
}
rownames(MSEs) <- cv$lambda
lambda.min <- as.numeric(names(which.min(rowMeans(MSEs))))
# get outputs; note that lambda.1se gives sparser outputs; near identical output to manual approach
# elastic net gives near similar results so we don't try it
coef(cv, s = cv$lambda.min)
modelL <- coef(cv, s = cv$lambda.1se)

## let's do a fractional logistic regression
tdf <- read.csv("/Users/martrinmunoz/Desktop/EconPredoc/Writing Samples/TdF/tdf_cleaned.csv")
vars_to_remove <- c('gen_ad_test')
tdf <- tdf[, !(colnames(tdf) %in% vars_to_remove)]
model1_remove <- c('avg_speed', 'num_entrants', 'first_prize_money', 
                   'tot_time_winner', 'year')
tdf <- tdf[, !(colnames(tdf) %in% model1_remove)]
logistic1 <- glm(ped_tot~., data = tdf, family = quasibinomial('logit'))
se_glm_robust_quasi = coeftest(logistic1, vcov = vcovHC(logistic1, type="HC1"))
se_glm_robust_quasi
summary(logistic1)
par(mfrow = c(2, 2))
plot(logistic1)
# now let's do a beta regression
logistic2 <- betareg(ped_tot~., data = tdf)
summary(logistic2)
plot(logistic2)
summary(tdf$ped_tot)


##below this are attempts at ridge regression, elastic net, and detrending data



#let's try ridge regression
ridge_cv <- cv.glmnet(x = X_train,
                      y = Y_train,
                      alpha = 0,
                      type.measure = 'mse',
                      nfolds = 5,
                      standardize = TRUE)

coef(ridge_cv, s = ridge_cv$lambda.1se)


# start with a plot of ped_total by year
tdf <- read.csv("/Users/martrinmunoz/Desktop/EconPredoc/Writing Samples/TdF/tdf_cleaned.csv")
ggplot(tdf, aes(x=year, y=ped_tot)) + geom_line()
# run regular linreg
model1 <- lm(ped_tot~., data = tdf)
summary(model1)
##demean data, but it ended up outputting everything as 0?
demean <- colwise(function(x) if(is.numeric(x)) x - mean(x) else x)
tdf_2 <- ddply(tdf, .(colnames(tdf)), demean)
tdf_2
ggplot(tdf_2, aes(x=year, y=ped_tot)) + geom_line()

#detrend data on order 1
detrend_1 <- detrend(tdf$ped_tot, p = 1)
tdf_2 <- data.frame(year = tdf$year, ped_tot = detrend_1, tdf[,3:ncol(tdf)])
ggplot(tdf_2, aes(x=year, y=ped_tot)) +
  geom_line()
#run linreg on detrended data (order = 1)
model2 <- lm(ped_tot~., tdf_2)
summary(model2)
# load data again
tdf <- read.csv("/Users/martrinmunoz/Desktop/EconPredoc/Writing Samples/TdF/tdf_cleaned.csv")
#detrend data on order 2
tdf$ped_tot <- detrend(tdf$ped_tot, p = 2)
#plot ped_tot vs. year
ggplot(tdf, aes(x=year, y=ped_tot)) +
  geom_line()
# remove gen_ad
tdf <- tdf[, !colnames(tdf) %in% 'gen_ad_test']
# vectorize indicator variables
indicator_var <- tdf %>% select(amph_test, epo_test, bio_passport, night_test,
                                ooct)
# scale tdf but without indicator variables

tdf_scaled <- tdf %>% 
  select(-colnames(indicator_var)) %>% 
  scale(.) %>% 
  cbind(., indicator_var)
  
# run linreg on detrended ped_tot and standardized variables
model3 <- lm(ped_tot~., tdf_scaled)
summary(model3)
coefs <- coef(model3)
coef_df <- data.frame(coef_names = names(coefs), coef_value = coefs)
coef_df %>% 
  arrange(desc(coef_value))


# restart and do stepwise regression
tdf <- read.csv("/Users/martrinmunoz/Desktop/EconPredoc/Writing Samples/TdF/tdf_cleaned.csv")
library(MASS)
install.packages('caret')
library(caret)
install.packages('leaps')
library('leaps')
#fit full model
model4 <- lm(ped_tot~., data = tdf)
#stepwise regression both (it picked backwards)
step.model <- stepAIC(model4, direction = "both", trace = FALSE)
summary(step.model)
#stepwise regression forward
step.model1 <- stepAIC(model4, direction = "forward", trace = FALSE)
summary(step.model1)
#stepwise regression backwards
step.model2 <- stepAIC(model4, direction = "backward", trace = FALSE)
summary(step.model2)
#stepwise backwawrds regression using caret
# Set up repeated k-fold cross-validation
train.control <- trainControl(method = "cv", number = 10)
# Train the model using caret stepwise regression
# first set up cv 
train.control <- trainControl(method = "cv", number = 10)
##run model
step.model3 <- train(ped_tot ~., data = tdf,
                    method = "leapBackward", 
                    tuneGrid = data.frame(nvmax = 1:5),
                    trControl = train.control
)
step.model3$results
summary(step.model3$finalModel)
coef(step.model3$finalModel, 5) 

# now we try elastic net
# start with a clean dataset
tdf <- read.csv("/Users/martrinmunoz/Desktop/EconPredoc/Writing Samples/TdF/tdf_cleaned.csv")
# remove gen_ad
tdf <- tdf[, !colnames(tdf) %in% 'gen_ad_test']
#set x and Y
X_train <- model.matrix(ped_tot~., data = tdf)[,-1]
Y_train <- tdf$ped_tot
#we need to tune the hyperparameter a
alpha <- list()
MSEs1 <- list()
for(i in 0:20){
  
  en_cv <- cv.glmnet(x = X_train, y = Y_train, alpha=i/20, type.measure = 'mse',
              nfolds=5, family = 'gaussian', standardize = TRUE)  
  alpha <- rbind(alpha, i/20)
  MSEs1 <- rbind(MSEs1, cv$cvm)
  
}
en_cv$alpha <- alpha
en_cv$lambda.min
summary(en_cv)
coef(en_cv, s=en_cv$lambda.min)
coef(en_cv, s= en_cv$lambda.1se)

