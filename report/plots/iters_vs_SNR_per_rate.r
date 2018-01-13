library(reshape2)
library(ggplot2)
library(scales)
library(extrafont)
library(gridExtra)
library(latex2exp)
library(readr)
library(data.table)

source('report/plots/utils.r')

data <- read_csv('results/SNRvsPe.csv.gz')

## count errors and number of message passing iteration
## per (n, rate, SNR) configuration, keeping
## "time per word" information as well
error_detection <- data.table(data)[,list(iterations = mean(iterations),
                                         errors = mean(errors)),
                                   by = 'n,rate,SNR']

p <- ggplot(data = error_detection[error_detection$errors > 1e-4,],
           mapping = aes(x = SNR,
                         y = iterations,
                         color = rate,
                         group = rate)) +
  ## axis and legend labels
  xlab(TeX('$E_b / N_o$ \\[dB\\]')) +
  ylab('Iterations') +
  labs(colour = 'Code rates') +

  geom_line() +
  scale_colour_brewer(palette="Spectral") +
  facet_wrap(~ n, nrow = 3) +
  ## scale_y_log10(breaks = 10^seq(10, -10),
  ##               labels = trans_format('log10', math_format(10^.x))) +
  scale_x_continuous(#breaks = unique(data$SNR),
                     labels = function(x) round(x, digits=2)) +
  mytheme +
  theme(
    legend.key.width = unit(1.5, 'cm'),
    legend.justification = c('left', 'top'),
    legend.position=c(0.78, 0.28),
  )

ggsave(plot = p + theme(plot.background = element_rect(
                          fill = 'transparent',
                          colour=NA)),
       filename = 'report/figures/iters_vs_SNR_per_rate.eps',
       width = 24 * 0.8,
       height = 17 * 0.8,
       unit = 'cm',
       device = 'eps',
       bg = 'transparent')

## p + theme(plot.background=element_rect(fill = 'black'))
