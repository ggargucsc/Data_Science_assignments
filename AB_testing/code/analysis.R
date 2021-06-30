setwd('~/Documents/buzzfeed/')
source("functions.r")

library(jsonlite)
library(dplyr)
library(dplyr)
library(ggplot2)
library(pscl)
library(car)

# load data
data = stream_in(file("shopping", open="r"))

# convert data types
data$shares = as.integer(data$shares)
data$ad_clicks = as.integer(data$ad_clicks)
data$bottom_of_page_reached = as.integer(data$bottom_of_page_reached)
data$bottom_related_link_clicks = as.integer(data$bottom_related_link_clicks)
data$top_related_link_clicks = as.integer(data$top_related_link_clicks)
data$shopping_pageviews = as.integer(data$shopping_pageviews)
data$total_pageviews = as.integer(data$total_pageviews)
data$date = as.Date(data$date, format = "%Y%m%d")
data$userid = factor(data$userid)
data$variant = factor(data$variant)
data$device_type = factor(data$device_type)

########################## analysis ############################
# check the sample size in each variant and device type
(data %>% group_by(variant, device_type) %>% 
  summarise(total_users = n(), 
            distint_users = n_distinct(userid), 
            perc_distinct = distint_users/total_users))

# Exclude same users from the data 
# by only keeping their first visit on website
data = as.data.frame(data %>% group_by(userid) %>% arrange(date) %>% slice(1))

# view descriptive statistics by variant and device type
# for shopping page views and total shares
library(plyr)
ddply(data, ~ variant + device_type, function(data) summary(data$shopping_pageviews))
ddply(data, ~ variant + device_type, summarise, Shopping_pages.mean=mean(shopping_pageviews), Shopping_pages.sd=sd(shopping_pageviews))
ddply(data, ~ variant + device_type, summarise, shares.mean=mean(shares), shares.sd=sd(shares))

# plot mean of shopping page views for each day in both the groups
shopping_pageviews = as.data.frame(data %>% 
                                     group_by(date, variant, device_type) %>% 
                                     summarise(m = mean(shopping_pageviews), sd = sd(shopping_pageviews), se=sd/sqrt(n())))

# plot showing mean of shopping page views for different devices
ggplot(shopping_pageviews[shopping_pageviews$device_type=='mobile',]) +
  theme_bw() +
  aes(x=date, y=m, ymin = m - se, ymax = m + se, color=variant, group=variant) +
  geom_line(linetype="dotted") + geom_point(size=3) +
  geom_point(position = position_dodge(width = 0.5)) + 
  ylab("mean of shopping page views") +
  ylim(0,1)

# shares distribution
shares = as.data.frame(data %>%  
                         group_by(variant, device_type) %>% 
                          summarise(m = mean(shares), sd = sd(shares), se = sd/sqrt(n())))

# plot showing mean of shares for different devices
ggplot(shares) + 
  theme_bw() + 
  aes(x = variant, y = m, ymin = m - se, ymax = m + se, color=device_type, group=device_type) + 
  geom_line(linetype="dotted") +
  geom_point(position = position_dodge(width = 0.5)) + 
  geom_errorbar(position = position_dodge(width = 0.5)) + 
  ylab("Total shares")
 

# plot some graphs in two groups
ggplot(shopping_pageviews) + 
  theme_bw() + 
  aes(x = variant, y = m, ymin = m - se, ymax = m + se, color=device_type, group=device_type) + 
  geom_line(linetype="dotted") +
  geom_point(position = position_dodge(width = 0.5)) + 
  geom_errorbar(position = position_dodge(width = 0.5)) + 
  ylab("Shopping pageviews")+
  ylim(0,1)

# boxplot
plot(shopping_pageviews ~ variant, data=data[data$device_type == 'mobile',])
plot(shopping_pageviews ~ variant, data=data[data$device_type == 'desktop',])

# how many people reached bottom of page in different variants?( around 60%)
bottom_reached = data %>%   
  mutate(bottom_reached_factor = ifelse(bottom_of_page_reached >=1, 'yes', 'no')) %>%
  group_by(variant, bottom_reached_factor) %>% summarise(count= n()) %>%
  mutate(perc = (count/sum(count))*100)

ggplot(bottom_reached,aes(variant,perc,fill=bottom_reached_factor))+
  geom_bar(alpha=0.5, stat="identity",position='dodge', width = 0.2) + 
  ylab("Percentage") + theme_bw()

# how many people who reached bottom of page clicked bottom module? ( only around 4%)
bottom_reached_clicked = data %>% filter(bottom_of_page_reached >=1) %>%
  mutate(bottom_clicked = ifelse(bottom_related_link_clicks >=1, 'yes', 'no')) %>%
  group_by(variant, bottom_clicked) %>% summarise(count = n()) %>%
  mutate(freq = (count / sum(count))*100) %>% filter(variant!='control')


ggplot(bottom_reached_clicked,aes(variant,freq,fill=bottom_clicked))+
  geom_bar(alpha=0.5, stat="identity",position='dodge', width = 0.2) + 
  ylab("Percentage") + theme_bw()


################################################################
################ Significance tests for mobile ################# 
mobile = as.data.frame(data %>% filter(device_type == 'mobile'))

# metrics are -- bottoms of pages reached, shopping pageviews,  
# ad clicks and shares

# let's check how the distribution of various metrics looks
plot_metrics(mobile[mobile$variant == 'control',]$shopping_pageviews)
plot_metrics(mobile[mobile$variant == 'bottom',]$total_pageviews)
plot_metrics(mobile[mobile$variant == 'top_and_bottom',]$bottom_of_page_reached)
plot_metrics(mobile[mobile$variant == 'control',]$ad_clicks)
plot_metrics(mobile[mobile$variant == 'control',]$shares)

# none of the metrics follows normal distribution
# Most of the values are zero, seems like a zero-inflated poisson distrib
# let's apply log tranformation and then check for normality
rand_nums = runif(nrow(mobile[mobile$variant == 'control',]), 0.01, 0.1)
plot_metrics(log(mobile[mobile$variant == 'control',]$shopping_pageviews + rand_nums))
plot_metrics(log(mobile[mobile$variant == 'control',]$total_pageviews))
plot_metrics(log(mobile[mobile$variant == 'control',]$bottom_of_page_reached + rand_nums))
plot_metrics(log(mobile[mobile$variant == 'control',]$ad_clicks + rand_nums))
plot_metrics(log(mobile[mobile$variant == 'control',]$shares + rand_nums))

mobile = as.data.frame(mobile %>% mutate(log_total_pageviews = log(total_pageviews)))
check_residuals_normality(y='log_total_pageviews', factor='variant', mobile)
# even after the transformations, it seems the new metrics do not follow normal distribution

# tests for homoscedasticity (homogeneity of variance)
library(car)
leveneTest(shopping_pageviews ~ variant, data=mobile, center=mean) # Levene's test
leveneTest(shopping_pageviews ~ variant, data=mobile, center=median) # Brown-Forsythe test

# zero inflated poisson 
# set sum-to-zero contrasts for the Anova call
contrasts(mobile$variant) <- "contr.sum"

model_pois = glm(shopping_pageviews ~ variant, data = mobile, family="poisson")
model_zero_pois =  zeroinfl(shopping_pageviews ~ variant, data = mobile, dist="poisson")
model_zero_nb =  zeroinfl(shopping_pageviews ~ variant, data = mobile, dist = "negbin")
# test to compare different models
vuong(model_zero_nb,model_pois)
vuong(model_zero_nb,model_zero_pois)
# zero inflated negative binomial fits the data  
Anova(model_zero_nb, type=3)
qqnorm(residuals(model_zero_nb)); qqline(residuals(model_zero_nb)) 

# non-parametric tests
## Nonparametric equivalent of one-way ANOVA
# Kruskal-Wallis test
library(coin)
kruskal_test(shopping_pageviews ~ variant, data=mobile, distribution="asymptotic") # can't do exact with 3 levels
# statistically significant for shopping pageviews
kruskal_test(total_pageviews ~ variant, data=mobile, distribution="asymptotic") 
kruskal_test(log_total_pageviews ~ variant, data=mobile, distribution="asymptotic") 
kruskal_test(ad_clicks ~ variant, data=mobile, distribution="asymptotic") 
kruskal_test(shares ~ variant, data=mobile, distribution="asymptotic") 
kruskal_test(bottom_of_page_reached ~ variant, data=mobile, distribution="asymptotic") 

# manual post hoc Mann-Whitney U pairwise comparisons
cb.ec = wilcox.test(mobile[mobile$variant == "control",]$shopping_pageviews, mobile[mobile$variant == "bottom",]$shopping_pageviews, exact=FALSE)
ctb.py = wilcox.test(mobile[mobile$variant == "control",]$shopping_pageviews, mobile[mobile$variant == "top_and_bottom",]$shopping_pageviews, exact=FALSE)
btb.py = wilcox.test(mobile[mobile$variant == "bottom",]$shopping_pageviews, mobile[mobile$variant == "top_and_bottom",]$shopping_pageviews, exact=FALSE)
options(scipen = 999)
p.adjust(c(cb.ec$p.value, ctb.py$p.value, btb.py$p.value), method="holm") # significant difference in control group


# alternative approach is using PMCMR for nonparam pairwise comparisons
library(PMCMR)
posthoc.kruskal.conover.test(shopping_pageviews ~ variant, data=mobile, p.adjust.method="holm") # Conover & Iman (1979)
posthoc.kruskal.conover.test(bottom_of_page_reached ~ variant, data=mobile, p.adjust.method="holm") # Conover & Iman (1979)


##################################################################
################## Significance tests for Desktop ################# 
desktop = as.data.frame(data %>% filter(device_type == 'desktop'))

# metrics are -- bottoms of pages reached, shopping pageviews,  
# ad clicks and shares

# let's check how the distribution of various metrics looks
plot_metrics(desktop[desktop$variant == 'control',]$shopping_pageviews)
plot_metrics(desktop[desktop$variant == 'bottom',]$total_pageviews)
plot_metrics(desktop[desktop$variant == 'top_and_bottom',]$bottom_of_page_reached)
plot_metrics(desktop[desktop$variant == 'control',]$ad_clicks)
plot_metrics(desktop[desktop$variant == 'control',]$shares)

# none of the metrics follows normal distribution
# Most of the values are zero, seems like a zero-inflated poisson distrib
# let's apply log tranformation and then check for normality
rand_nums = runif(nrow(desktop[desktop$variant == 'control',]), 0.01, 0.1)
plot_metrics(log(desktop[desktop$variant == 'control',]$shopping_pageviews + rand_nums))
plot_metrics(log(desktop[desktop$variant == 'control',]$total_pageviews))
plot_metrics(log(desktop[desktop$variant == 'control',]$bottom_of_page_reached + rand_nums))
plot_metrics(log(desktop[desktop$variant == 'control',]$ad_clicks + rand_nums))
plot_metrics(log(desktop[desktop$variant == 'control',]$shares + rand_nums))

desktop = as.data.frame(desktop %>% mutate(log_total_pageviews = log(total_pageviews)))
check_residuals_normality(y='log_total_pageviews', factor='variant', desktop)
# even after the transformations, it seems the new metrics do not follow normal distribution

# tests for homoscedasticity (homogeneity of variance)
library(car)
leveneTest(shopping_pageviews ~ variant, data=desktop, center=mean) # Levene's test
leveneTest(shopping_pageviews ~ variant, data=desktop, center=median) # Brown-Forsythe test
leveneTest(shares ~ variant, data=desktop, center=median) 
# zero inflated poisson 
# set sum-to-zero contrasts for the Anova call
contrasts(desktop$variant) <- "contr.sum"

model_pois = glm(shopping_pageviews ~ variant, data = desktop, family="poisson")
model_zero_pois =  zeroinfl(shopping_pageviews ~ variant, data = desktop, dist="poisson")
model_zero_nb =  zeroinfl(shopping_pageviews ~ variant, data = desktop, dist = "negbin")
model_zero_nb_shares =  zeroinfl(shares ~ variant, data = desktop, dist = "negbin")

# test to compare different models
vuong(model_zero_nb,model_pois)
vuong(model_zero_nb,model_zero_pois)
# zero inflated negative binomial fits the data  
Anova(model_zero_nb, type=3)
qqnorm(residuals(model_zero_nb)); qqline(residuals(model_zero_nb)) 

# non-parametric tests
## Nonparametric equivalent of one-way ANOVA
# Kruskal-Wallis test
library(coin)
kruskal_test(shopping_pageviews ~ variant, data=desktop, distribution="asymptotic") # can't do exact with 3 levels
# statistically significant for shopping pageviews
kruskal_test(total_pageviews ~ variant, data=desktop, distribution="asymptotic") 
kruskal_test(log_total_pageviews ~ variant, data=desktop, distribution="asymptotic") 
kruskal_test(ad_clicks ~ variant, data=desktop, distribution="asymptotic") 
kruskal_test(shares ~ variant, data=desktop, distribution="asymptotic") 
kruskal_test(bottom_of_page_reached ~ variant, data=desktop, distribution="asymptotic") 

# manual post hoc Mann-Whitney U pairwise comparisons
cb.ec = wilcox.test(desktop[desktop$variant == "control",]$shopping_pageviews, desktop[desktop$variant == "bottom",]$shopping_pageviews, exact=FALSE)
ctb.py = wilcox.test(desktop[desktop$variant == "control",]$shopping_pageviews, desktop[desktop$variant == "top_and_bottom",]$shopping_pageviews, exact=FALSE)
btb.py = wilcox.test(desktop[desktop$variant == "bottom",]$shopping_pageviews, desktop[desktop$variant == "top_and_bottom",]$shopping_pageviews, exact=FALSE)
options(scipen = 999)
p.adjust(c(cb.ec$p.value, ctb.py$p.value, btb.py$p.value), method="holm") # significant difference in control group

# manual post hoc Mann-Whitney U pairwise comparisons
cb.ec = wilcox.test(desktop[desktop$variant == "control",]$shares, desktop[desktop$variant == "bottom",]$shares, exact=FALSE)
ctb.py = wilcox.test(desktop[desktop$variant == "control",]$shares, desktop[desktop$variant == "top_and_bottom",]$shares, exact=FALSE)
btb.py = wilcox.test(desktop[desktop$variant == "bottom",]$shares, desktop[desktop$variant == "top_and_bottom",]$shares, exact=FALSE)
options(scipen = 999)
p.adjust(c(cb.ec$p.value, ctb.py$p.value, btb.py$p.value), method="holm") # significant difference in control group


# alternative approach is using PMCMR for nonparam pairwise comparisons
library(PMCMR)
posthoc.kruskal.conover.test(shopping_pageviews ~ variant, data=desktop, p.adjust.method="holm") # Conover & Iman (1979)
posthoc.kruskal.conover.test(shares ~ variant, data=desktop, p.adjust.method="holm") # Conover & Iman (1979)
posthoc.kruskal.conover.test(bottom_of_page_reached ~ variant, data=desktop, p.adjust.method="holm") # Conover & Iman (1979)


######################################################################
################## optimal poistion of the unit #######################
detach("package:plyr", unload=TRUE) 
data %>% filter(variant=='top_and_bottom', device_type=='desktop') %>% group_by(top_related_link_clicks) %>%
  summarise(count = n()) %>% mutate(freq = (count/sum(count))*100)
data %>% filter(variant=='top_and_bottom', device_type=='desktop') %>% group_by(bottom_related_link_clicks) %>%
  summarise(count = n()) %>% mutate(freq = (count/sum(count))*100)


# We can also set test the results as within group study
hist(data[data$variant=='top_and_bottom' & data$device_type=='desktop',]$bottom_related_link_clicks)


# non-parametric test
# Wilcoxon signed-rank test on clicks
library(coin)
index_desktop = data$variant=='top_and_bottom' & data$device_type == 'desktop'
wilcox.test(data[index,]$bottom_related_link_clicks, data[index,]$top_related_link_clicks, paired = TRUE)

index_mobile = data$variant=='top_and_bottom' & data$device_type == 'mobile'
wilcox.test(data[index_mobile,]$bottom_related_link_clicks, data[index_mobile,]$top_related_link_clicks, paired = TRUE)
