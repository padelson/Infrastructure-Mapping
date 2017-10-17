library(ggmap)
library(ggplot2)

csv_path = "/Users/anniehu/Desktop/CS325B/"
water_interruption = "bl_dw19"
water_days = "bl_dw20"
water_satisfaction = "bl_dw25"
garbage_collection = "bl_dw46"
garbage_frequency = "bl_dw47"
electricity = "bl_dw56"
electricity_days = "bl_dw63"
healthcare_use = "bl_sd33"
healthcare_distance = "bl_sd35"
healthcare_satisfaction = "bl_sd36"
education_use = "bl_sd43"
education_distance = "bl_sd46"
education_satisfaction = "bl_sd51"

data_raw = read.csv(paste(csv_path, "Addis_data.csv", sep=""))

d <- data.frame(lat=data_raw[['bl_bi24latitude']],
                lon=data_raw[['bl_bi24longitude']],
                interruption=data_raw[[education_use]],
                days=data_raw[[education_distance]],
                satisfaction=data_raw[[education_satisfaction]])

Addis <- get_map(location=c(left = 34, bottom = 5, right =
                              44, top = 13), maptype='hybrid',
                            source='google', zoom=9)

p <- ggmap(Addis)
p <- p + geom_point(data=d[d$interruption==1,], aes(x=lon, y=lat), color="green")
p <- p + geom_point(data=d[d$interruption==2,], aes(x=lon, y=lat), color="red")
p <- p + geom_point(data=d[d$days < 0,], aes(x=lon, y=lat), color="gray")
p <- p + geom_point(data=d[d$days >= 0,], aes(x=lon, y=lat, color=days)) + scale_color_gradient(low='white',high='red')
p <- p + geom_point(data=d[d$satisfaction>=0,], aes(x=lon, y=lat, color=satisfaction)) + scale_color_gradient(low='white',high='red')
p

