library(ggmap)
library(ggplot2)

csv_path = "/Users/anniehu/Desktop/CS325B/"

data_raw = read.csv(paste(csv_path, "Addis_data.csv", sep=""))

d <- data.frame(lat=data_raw[['bl_bi24latitude']],
                lon=data_raw[['bl_bi24longitude']])

Addis <- get_map(location=c(left = 34, bottom = 5, right =
                              44, top = 13), maptype='satellite',
                            source='google', zoom=9)

p <- ggmap(Addis)
p <- p + geom_point(data=d, aes(x=lon, y=lat))
p
