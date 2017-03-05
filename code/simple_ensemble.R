setwd("~/kobe")
preds_xgb <- read.csv("submission/sub_xgb.csv")
preds_tf <- read.csv("submission/sub_tf.csv")

preds <- preds_xgb
preds$shot_made_flag <- preds_xgb$shot_made_flag * 0.7 + preds_tf$shot_made_flag * 0.3

write.csv(preds, "sub_xgb_tf.csv", row.names=FALSE)
