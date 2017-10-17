library(ggmap)
library(ggplot2)

csv_path = "/Users/anniehu/Desktop/CS325B/"

water_interruption = "bl_dw19"
water_days = "bl_dw20"


data_raw = read.csv(paste(csv_path, "Addis_data.csv", sep=""))

d <- data.frame(lat=data_raw[['bl_bi24latitude']],

                lon=data_raw[['bl_bi24longitude']],
                interruption=data_raw[[water_interruption]],
                days=data_raw[[water_days]])

Addis <- get_map(location=c(left = 34, bottom = 5, right =
                              44, top = 13), maptype='hybrid',
                            source='google', zoom=9)

p <- ggmap(Addis)
p <- p + geom_point(data=d[d$interruption==2,], aes(x=lon, y=lat), color="green")
p <- p + geom_point(data=d[d$days < 0,], aes(x=lon, y=lat), color="gray")
p <- p + geom_point(data=d[d$days >= 0,], aes(x=lon, y=lat, color=days)) + scale_color_gradient(low='white',high='red')
p

