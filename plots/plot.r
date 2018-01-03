library(reshape2)
library(ggplot2)
library(scales)
library(extrafont)
library(gridExtra)
library(latex2exp)
library(readr)
library(data.table)

source('plots/utils.r')

data <- read_csv('results/SNRvsPe.csv.gz')

## count errors and number of message passing iteration
## per (n, rate, SNR) configuration, keeping
## "time per word" information as well
error_detection <- data.table(data)[,list(iters = mean(iterations),
                                         errors = mean(errors)),
                                   by = 'n,rate,SNR,`time per word`']

p <- ggplot(data = error_detection[error_detection$errors != 0,],
           mapping = aes(x = SNR,
                         y = errors,
                         color = n,
                         group = n)) +
  ## axis and legend labels
  xlab(TeX('$E_b / N_o$ \\[dB\\]')) +
  ylab('Packet Error Rate') +
  labs(colour = 'Code lengths') +

  geom_line() +
  facet_wrap(~ rate, nrow = 2) +
  scale_y_log10(breaks = 10^seq(10, -10),
                labels = trans_format('log10', math_format(10^.x))) +
  scale_x_continuous(breaks = unique(data$SNR),
                     labels = function(x) round(x, digits=2)) +
  theme(
    rect = element_rect(fill = 'black'),
    panel.background = element_rect(fill = 'transparent', colour = NA),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(colour='grey10'),

    axis.text = element_text(colour = 'grey50'),
    axis.text.x = element_text(angle = 45, hjust = 1),

    legend.background = element_rect(fill = 'transparent', colour = NA),
    legend.title = element_text(colour = 'grey60',
                                size = rel(1.1),
                                vjust=1),
    legend.text = element_text(colour  ='grey60'),

    strip.background = element_rect(fill = 'grey10', colour = NA),
    strip.text = element_text(colour = 'grey50'),

    axis.title = element_text(colour = 'grey60', size = rel(1.1)),
    )



## ggsave(plot = p,
##        filename = 'plots/figures/SNRvsPe.png',
##        width = 10,
##        height = 8,
##        unit = 'cm',
##        device = 'png',
##        bg = 'transparent')

p
