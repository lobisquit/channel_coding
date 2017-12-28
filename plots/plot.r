library(reshape2)
library(ggplot2)
library(scales)
library(extrafont)
library(gridExtra)

source('plots/utils.r')

data <- read.csv('results/SNRvsPe.csv.gz')

## count errors and number of message passing iteration
## per (n, rate, SNR) configuration, keeping
## "time per word" information as well
error_detection <- aggregate(
  cbind(iterations, errors) ~ n + rate + SNR + time.per.word, data, mean)

p <- ggplot(data = error_detection[error_detection$errors != 0,],
           mapping = aes(x = SNR,
                         y = time.per.word,
                         color = n,
                         group = n)) +
  geom_line() +
  scale_y_log10(breaks = 10^seq(0, -5),
                labels = trans_format('log10', math_format(10^.x))) +
  scale_x_continuous(breaks = unique(data$SNR),
                     labels = function(x) round(x, digits=2)) +

  facet_wrap(~ rate, ncol = 2) +
  my_theme() +
  theme(
    plot.background = element_rect(fill = 'transparent')
  )

## ggsave(plot = p,
##        filename = 'plots/figures/SNRvsPe.png',
##        unit = 'cm',
##        device = 'png',
##        bg = 'transparent')

p
