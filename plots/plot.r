library(reshape2)
library(ggplot2)
library(scales)
library(extrafont)
library(gridExtra)

source('plots/utils.r')

data <- read.csv('results/SNRvsPe.csv')

data$Pe <- data$Perror + data$Pfailure

p <- ggplot(data = data[data$Pe != 0,],
           mapping = aes(x = SNR,
                         y = Pe,
                         color = n,
                         group = n)) +
  geom_line() +
  scale_y_log10() + ## this needs data$Pe != 0
  scale_x_continuous(breaks = unique(data$SNR)) +

  ## highlight "always correct" and "always bad" points
  ## geom_point(data = data[data$Pe == 0,], colour = "green") +
  ## geom_point(data = data[data$Pe == 1,], colour = "red") +

  facet_wrap(~ rate, ncol = 2) +
  my_theme()

## ggsave(plot=p,
##        filename='plots/figures/SNRvsPe.png',
##        unit='cm',
##        device='png')

p
