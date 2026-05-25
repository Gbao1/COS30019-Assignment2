Question: If you know the traffic flow on a link (e.g., number of vehicles per hour), how
do you estimate the travel time on that link?
When calculating the travel time, the formula time = distance / speed is used for each segment
of the route (i.e., each link) and the time delay for each intersection is added to this result.
When calculating the time for each segment a different speed is used, this speed is affected by
the predicted traffic flow.
When calculating the speed of the SCATS site, we’ll make the following assumption:
● Between two SCATS sites, the flow is calculated from the starting SCATS site.
Using traffic flow prediction a Traffic model was created for the conversion between traffic flow
and speed. This is to determine the speed between each SCATS site. More information about
this can be found here: https://en.wikipedia.org/wiki/Fundamental_diagram_of_traffic_flow
Figure 1: graph function that converts from flow to speed
Note: The above quadratic equation is a (over)simplified version of the fundamental diagram
but we’ll use it for the purpose of Assignment 2B.
As the relationship between speed and flow is parabolic there are two speeds for every flow rate
as distinguished by the green and red lines on the graph, thus it is broken up into two functions.
The green line represents the equation when the road is under capacity and the red line
represents the speed when the road is over capacity.
The turning point of the parabola corresponds to the road's flow and speed at capacity. The
values of 1500 vehicles/hour and 32 km/hr do not represent the flow of the actual traffic network
but were chosen as it effectively demonstrates how traffic conditions affect the routing.
The blue dashed line is the speed limit, so when the flow is at or below 351 vehicles per hour
the expected speed is above the speed limit so the speed is capped at 60, then when the flow
is above 351 vehicles per hour the traffic is high and there are too many cars on the road to
drive at the speed limit so the expected speed is used in the time calculation.
When producing this graph some more assumptions are made:
● Each road in the network had the same speed and flow at capacity
● The traffic at each segment is under capacity.
To summarise, the relationship between flow and speed can be captured by the following function:
flow = -1.4648375*(speed)2 + 93.75*(speed)
When estimating the speed from the predicted flow, you’ll need to determine whether the flow falls
into the red (congested road) or green (number of vehicles is still under the road capacity) part of
the curve and compute the corresponding speed.