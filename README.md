# bike_uber_manhattan

Check out the medium post about this project: []

### Description

Welcome to my Manhattan Mobility Study! We are going to take dive into the Big Apple and try to get a better understanding of two of its forms of transportation: a Bike Sharing System and Uber Rides.

How are each of them used throughout the day? Is there a big difference in their use between weekends and week days? Which are the favorite districts for cyclists? How is the dynamic of a neighborhood very popular among tourists? What about local workers?

We are going to try to answer some of these questions using python, with a little help from folium, a very powerful library that will help us make some beautiful maps!


### Key Takeaways

- The southwest half of Manhattan have a lot of districs with very popular among bike and uber users. In the other half, the volume of trips drops a lot.
- Even though we can spot some difference, in general, popular districs among uber users are also popular among bike users.
- The bike system is used a lot by local workers. We can see a very high number of trips starting in peak hours in the weekend. This effect is not very clear for uber rides.
- We can see a lot of trips starting at transport hubs, like the Penn Station and the Grand Central Terminal. That makes a lot of sense, since there is a lot of people that goes to the city but don't live in Manhattan.
- For both modes, the number of trips in a weekday is a little higher than in the weekend.
- We can see a very high demand for uber rides at weekend nights. Those trips start mostly in the southern districts, such as west village and east village.



### Files in the repository:


- 201809-citibike-tripdata.csv*: The csv file with all the bike trips, collected from the city bike website
- uber-raw-data-sep14.csv*: The csv file with all the uber trips, compiled by the Five-Thirty-Eight portal
- nyc_map: A folder with a lot of files, that contains a shapefile that will allow us to draw the city neighborhoods limits.
- uber_bike_manhattan_study.ipynb: The notebook of the project
- uber_bike_manhattan_study.py: The project as a python file
- uber_bike_manhattan_study.html: The project as a html file
- bike_week, bike_weekend, uber_week, uber_weekend: the folders that contains the gif of the timelapse of the trips, and all the auxiliar .png images

Important Note: The csv files used on this project, (with uber and bike trips) are to bigger than github allows. Therefore, please download it on the original source:

[Bike Trips](https://s3.amazonaws.com/tripdata/index.html)

[Uber Trips](https://github.com/fivethirtyeight/uber-tlc-foil-response)

### Libraries used:
Pandas, Numpy, Datetime, Seaborn, Matplotlib, Plotly, Branca, Folium, Geopandas, Imageio, Selenium

Thanks for reading!
