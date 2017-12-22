library(reshape2)
library(ggplot2)
library(scales)
library(extrafont)

my_theme = function() {
  theme(
    ## set everything to serif, we are good people after all
    text = element_text("Charter"),

    ## axis title size
    axis.title.x = element_text(size = 12, margin = margin(t = 15)),
    axis.title.y = element_text(size = 12, margin = margin(r = 15)),

    ## set subtitles preferences, when facet griding
    strip.background = element_rect(fill="white"),
    strip.text = element_text(size = 10),

    ## plot area setting
    panel.background = element_rect(fill = "#E6E6E6"),
    panel.grid.major = element_line(colour = "#666666", size = 0.1),
    panel.grid.minor = element_blank(),

    ## legend settings: lines background, title
    legend.key = element_blank(),
    legend.title = element_blank(),
    legend.text = element_text(size = 10),
    )
  }
