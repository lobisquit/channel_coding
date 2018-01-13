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
                         color = n,
                         group = n)) +
  ## axis and legend labels
  xlab(TeX('$E_b / N_o$ \\[dB\\]')) +
  ylab('Iterations') +
  labs(colour = 'Code lengths') +

  geom_line() +
  scale_colour_distiller(palette = "Spectral") +
  facet_wrap(~ rate, nrow = 2) +
  scale_x_continuous(#breaks = unique(data$SNR),
                     labels = function(x) round(x, digits=2)) +
  mytheme

ggsave(plot = p + theme(plot.background = element_rect(
                          fill = 'transparent',
                          colour=NA)),
       filename = 'report/figures/iters_vs_SNR_per_length.eps',
       width = 24 * 0.8,
       height = 17 * 0.8,
       unit = 'cm',
       device = 'eps',
       bg = 'transparent')

## p + theme(plot.background=element_rect(fill = 'black'))
