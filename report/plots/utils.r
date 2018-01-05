library(reshape2)
library(ggplot2)
library(scales)
library(extrafont)
library(gridExtra)

mytheme <- theme(
  ## rect = element_rect(fill = 'black'),
  panel.background = element_rect(fill = 'transparent', colour = NA),
  panel.grid.minor = element_blank(),
  panel.grid.major = element_line(colour='grey10'),

  axis.text = element_text(colour = 'grey50'),
  axis.text.x = element_text(angle = 45, hjust = 1),

  legend.background = element_rect(fill = 'transparent', colour = NA),
  legend.title = element_text(colour = 'grey60',
                              size = rel(1.1),
                              vjust=1),
  legend.text = element_text(colour  ='grey60', size = rel(0.8)),
  legend.key = element_rect(fill='transparent', colour=NA),

  strip.background = element_rect(fill = 'grey10', colour = NA),
  strip.text = element_text(colour = 'grey50'),

  axis.title = element_text(colour = 'grey60', size = rel(1.1)),
  )
