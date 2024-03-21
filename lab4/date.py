import datetime as dt
from datetime import datetime

#ex1
now = datetime.datetime.now()
ans = now - datetime.timedelta(5)
print(ans)

#ex2
now = dt.datetime.now()
yest = now - dt.timedelta(1)
tomor = now + dt.timedelta(1)
print("Yesterday was ", yest)
print("Today is ", now)
print("Tomorrow will be ", tomor)

#ex3
now = datetime.datetime.now().replace(microsecond=0)
print(now)

#ex4
date1 = datetime(2023, 5, 15, 12, 30, 0)  
date2 = datetime(2023, 5, 10, 8, 45, 0)   
minus = date1 - date2
diff = minus.total_seconds()
print(diff)