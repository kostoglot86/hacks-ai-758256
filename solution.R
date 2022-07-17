library(tidyverse)
library(dummies)
library(lightgbm)
library(lubridate)
library(caret)
library(data.table)
library(MLmetrics)

# Parameters for final model
param_lgb= list(objective = "multiclass",
                max_bin = 256,
                learning_rate = 0.03,
                num_leaves = 127,
                bagging_fraction = 0.7,
                feature_fraction = 0.7,
                min_data = 50,
                bagging_freq = 1,
                metric = "multi_logloss",
                num_class = 6)

# Should be changed to actual path with datasets
PATH_TO_DATA = ""

train = read.csv(paste0(PATH_TO_DATA, "train_dataset_train.csv"), header = T)
test = read.csv(paste0(PATH_TO_DATA, "test_dataset_test.csv"), header = T)

target = train$target
tr_te = rbind(train[c(1:58)],test)

# Preprocessing data and feature generation
tr_te$month_id = as.Date(tr_te$month_id, format = "%m/%d/%Y")
tr_te$carts_created_at = as.Date(tr_te$carts_created_at, format = "%m/%d/%Y")
tr_te$delta = as.numeric(tr_te$month_id - tr_te$carts_created_at)
tr_te$month_id = as.numeric(tr_te$month_id)
tr_te$carts_created_at = as.numeric(tr_te$carts_created_at)

tr_te = tr_te %>% group_by(student_id) %>% mutate(cnt_unique_prog_by_stud = length(unique(program_id)),
                                                  cnt_unique_month = length(unique(month_id))) %>% data.frame()

dop_f = unique(tr_te[c('student_id', 'carts_created_at')]) %>% arrange(carts_created_at) %>% 
  group_by(student_id) %>% mutate(prev_carts = carts_created_at - lag(carts_created_at),
                                  next_carts = lead(carts_created_at) - carts_created_at)

tr_te = left_join(tr_te, dop_f)

tr_te$magic = tr_te$delta - tr_te$next_carts

tr_te = tr_te %>% group_by(program_id) %>% mutate(cnt_unique_stud_by_prog = length(unique(student_id)),
                                                  avg_price_by_prog = mean(price, na.rm = T),
                                                  avg_age_by_prog = mean(age_indicator, na.rm = T)) %>% data.frame()

tr_te = tr_te %>% group_by(student_id) %>% mutate(min_price = min(price),
                                                  max_price = max(price),
                                                  min_carts = min(carts_created_at),
                                                  max_carts = max(carts_created_at)) %>% data.frame()

tr_te = tr_te %>% group_by(age_indicator) %>% mutate(avg_price_by_age = mean(price, na.rm = T)) %>% data.frame()
tr_te = tr_te %>% group_by(student_id) %>% mutate(mean_spent_time = mean(spent_time_total, na.rm = T),
                                                  min_spent_time = min(spent_time_total, na.rm = T),
                                                  max_spent_time = max(spent_time_total, na.rm = T)) %>% data.frame()

tr_te = tr_te %>% group_by(program_id) %>% mutate(mean_spent_time_by_prog = mean(spent_time_total, na.rm = T),
                                                  min_spent_time_by_prog = min(spent_time_total, na.rm = T),
                                                  max_spent_time_by_prog = max(spent_time_total, na.rm = T)) %>% data.frame()

dop_f2 = tr_te %>% group_by(student_id) %>% summarize(mean_feedback1 = mean(feedback_avg_d1, na.rm = T),
                                                      min_feedback1 = min(feedback_avg_d1, na.rm = T),
                                                      max_feedback1 = max(feedback_avg_d1, na.rm = T),
                                                      mean_feedback2 = mean(feedback_avg_d2, na.rm = T),
                                                      min_feedback2 = min(feedback_avg_d2, na.rm = T),
                                                      max_feedback2 = max(feedback_avg_d2, na.rm = T),
                                                      mean_feedback3 = mean(feedback_avg_d3, na.rm = T),
                                                      min_feedback3 = min(feedback_avg_d3, na.rm = T),
                                                      max_feedback3 = max(feedback_avg_d3, na.rm = T),
                                                      mean_feedback4 = mean(feedback_avg_d4, na.rm = T),
                                                      min_feedback4 = min(feedback_avg_d4, na.rm = T),
                                                      max_feedback4 = max(feedback_avg_d4, na.rm = T),
                                                      mean_feedback5 = mean(feedback_avg_d5, na.rm = T),
                                                      min_feedback5 = min(feedback_avg_d5, na.rm = T),
                                                      max_feedback5 = max(feedback_avg_d5, na.rm = T)) %>% data.frame()

tr_te = left_join(tr_te, dop_f2)     

tr_te = tr_te %>% group_by(student_id) %>% mutate(sum_spent_time = sum(spent_time_total, na.rm = T),
                                                  sum_activity   = sum(activity, na.rm = T),
                                                  sum_lessons    = sum(lessons, na.rm = T),
                                                  sum_webinars   = sum(webinars, na.rm = T),
                                                  sum_m_total_calls    = sum(m_total_calls, na.rm = T),
                                                  sum_p_total_duration = sum(p_total_duration, na.rm = T)) %>% data.frame()

tr_te = tr_te %>% group_by(city) %>% mutate(stud_by_city = length(unique(student_id))) %>% data.frame()


tr_te = tr_te %>% group_by(carts_created_at) %>% mutate(stud_by_carts = length(unique(student_id)))

tr_te = tr_te %>% group_by(student_id) %>% mutate(nan_activity_months = sum(is.na(activity) == T),
                                                  nan_activity_share = sum(is.na(activity) == T)/length(activity))

tr_te = tr_te %>% group_by(student_id, month_id) %>% mutate(prog_cnt_by_stud_in_month = length(unique(program_id)))

tr_te = tr_te %>% group_by(program_id) %>% mutate(max_price_by_prog = max(price, na.rm = T),
                                                  min_price_by_prog = min(price, na.rm = T)) %>% data.frame()

dop_f3 = tr_te %>% group_by(student_id) %>% summarize(time_from_bought_d1 = min(month_id[bought_d1 == 1]),
                                                      time_from_bought_d2 = min(month_id[bought_d2 == 1]),
                                                      time_from_bought_d3 = min(month_id[bought_d3 == 1]),
                                                      time_from_bought_d4 = min(month_id[bought_d4 == 1]),
                                                      time_from_bought_d5 = min(month_id[bought_d5 == 1])) %>% data.frame()

tr_te = left_join(tr_te, dop_f3)
tr_te$time_from_bought_d1 = tr_te$month_id - tr_te$time_from_bought_d1
tr_te$time_from_bought_d2 = tr_te$month_id - tr_te$time_from_bought_d2
tr_te$time_from_bought_d3 = tr_te$month_id - tr_te$time_from_bought_d3
tr_te$time_from_bought_d4 = tr_te$month_id - tr_te$time_from_bought_d4
tr_te$time_from_bought_d5 = tr_te$month_id - tr_te$time_from_bought_d5

for (i in c(as.numeric(which(sapply(tr_te, "class") == 'character'))))
  tr_te[,i] = as.numeric(as.factor(tr_te[,i]))

train = tr_te[c(1:dim(train)[1]),]
te = tr_te[c((dim(train)[1] + 1):(dim(tr_te)[1])),]

set.seed(13)
train$fold <- createFolds(target, 1:nrow(train), k=5,list = FALSE)
fold.ids <- unique(train$fold)
custom.folds <- vector("list", length(fold.ids))
i <- 1
for( id in fold.ids){
  custom.folds[[i]] <- which(train$fold %in% id )
  i <- i+1
}

# Example of permutation importance selection (note that model_lgb1 shoul be trained on whole set of features)
# test_permute = train[train$fold == 1,][-c(exclude)]
# answers = target[train$fold == 1]
# perm_imp_res = c()
# for (i in c(6:dim(test_permute)[2]))
# {
#   message(i)
#   test_permute0 = test_permute
#   mll = 0
#   for (j in c(1:5)) {
#     set.seed(i + j)
#     test_permute0[,i] = sample(test_permute0[,i])
#     pred = predict(model_lgb1, as.matrix(test_permute0))
#     mll_iter = MultiLogLoss(matrix(data = pred, nrow = 40000, ncol = 6, byrow = TRUE), answers)
#     mll = mll + mll_iter
#   }
#   metric = mll/5
#   print(metric)
#   df_iter = data.frame(i = i, feature = colnames(test_permute)[i], metric = metric)
#   perm_imp_res = rbind(perm_imp_res, df_iter)
# }

# Selected features after PI selection
include2 = c('delta', 'magic', 'carts_created_at', 'max_carts', 'student_id', 'nan_activity_share', 'avg_age_by_prog', 
             'mean_spent_time', 'age_indicator', 'stud_by_carts', 'max_price', 'program_id', 'sum_activity', 'sum_lessons', 'month_id', 
             'sum_webinars', 'min_carts', 'max_spent_time', 'cnt_unique_stud_by_prog', 'sum_spent_time', 'min_price', 'avg_price_by_age', 
             'mean_spent_time_by_prog', 'max_spent_time_by_prog', 'min_spent_time', 'cnt_unique_month', 'max_price_by_prog', 
             'support_feedback_avg', 'avg_price_by_prog', 'price', 'min_price_by_prog', 'mean_feedback1', 'prog_cnt_by_stud_in_month', 
             'nan_activity_months', 'ABC', 'cnt_unique_prog_by_stud', 'mean_feedback5', 'm_avg_talk_duration', 'browser', 'm_avg_duration', 
             'feedback_avg_d1', 'm_total_duration', 'mean_feedback4', 'mean_feedback3', 'gender', 'os', 'max_feedback4', 'mean_feedback2', 
             'communication_type', 'payment_type', 'max_feedback1', 'next_carts', 'promo', 'sum_m_total_calls', 'speed_recall', 
             'min_feedback1', 'auto_payment', 'max_feedback3', 'p_total_duration', 'feedback_avg_d4', 'min_feedback4', 'm_total_calls', 
             'sum_p_total_duration', 'platform', 'max_feedback2', 'feedback_avg_d3', 'feedback_avg_d2', 'm_was_conversations', 
             'feedback_avg_d5', 'max_feedback5', 'time_from_bought_d1', 'time_from_bought_d2', 'time_from_bought_d3', 'time_from_bought_d4', 
             'time_from_bought_d5')


# VALIDATION BLOCK (NOT NECESSARY FOR PREDICTION PHASE)
# dtrain <- lgb.Dataset(as.matrix(train[train$fold != 1,][c(include2)]),label = target[train$fold != 1])
# dtest = lgb.Dataset(as.matrix(train[train$fold == 1,][c(include2)]),label = target[train$fold == 1])
# valids = list(test = dtest)
# model_lgb1 = lgb.train(data=dtrain, valids = valids, params = param_lgb, nrounds=10000,
#                        eval_freq = 100, early_stopping_rounds = 200)
# 
# pr = predict(model_lgb1, as.matrix(train[train$fold == 1,][c(include2)]))
# prm = data.frame(matrix(data = pr, nrow = 40000, ncol = 6, byrow = TRUE))
# prm$target = apply(prm, 1, function(x) which.max(x) - 1)
# prm$true = target[train$fold == 1]
# 
# rec0 = sum(prm$target == 0 & prm$true == 0)/sum(prm$true == 0)
# rec1 = sum(prm$target == 1 & prm$true == 1)/sum(prm$true == 1)
# rec2 = sum(prm$target == 2 & prm$true == 2)/sum(prm$true == 2)
# rec3 = sum(prm$target == 3 & prm$true == 3)/sum(prm$true == 3)
# rec4 = sum(prm$target == 4 & prm$true == 4)/sum(prm$true == 4)
# rec5 = sum(prm$target == 5 & prm$true == 5)/sum(prm$true == 5)
# 
# prec0 = sum(prm$target == 0 & prm$true == 0)/sum(prm$target == 0)
# prec1 = sum(prm$target == 1 & prm$true == 1)/sum(prm$target == 1)
# prec2 = sum(prm$target == 2 & prm$true == 2)/sum(prm$target == 2)
# prec3 = sum(prm$target == 3 & prm$true == 3)/sum(prm$target == 3)
# prec4 = sum(prm$target == 4 & prm$true == 4)/sum(prm$target == 4)
# prec5 = sum(prm$target == 5 & prm$true == 5)/sum(prm$target == 5)
# 
# score = 0.8*(prec0 + prec1 + prec2 + prec3 + prec4 + prec5)/6 + 0.2*(rec0 + rec1 + rec2 + rec3 + rec4 + rec5)/6
# print(score)

# Prediction phase
dtrain <- lgb.Dataset(as.matrix(train[c(include2)]),label = target)
pr_final = 0
ITERS = 5
for (i in c(1:ITERS))
{
  message(i)
  model_lgb = lgb.train(data=dtrain, params = param_lgb, nrounds = 1300, bagging_seed = 13 + i, feature_fraction_seed = 42 + i)
  pr = predict(model_lgb, as.matrix(te[c(include2)]))
  pr_final = pr_final + pr
}

pr_res = pr_final/ITERS
pr_res = data.frame(matrix(data = pr_res, nrow = length(pr_res)/6, ncol = 6, byrow = TRUE))
pr_res$target = apply(pr_res, 1, function(x) which.max(x) - 1)

sub17 = data.frame(id = test$id, target = pr_res$target)
write.csv(sub17, paste0(PATH_TO_DATA, "sub17.csv"), row.names = F, quote = F)
