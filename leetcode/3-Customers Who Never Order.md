# [3. Customers Who Never Order](https://leetcode.com/problems/customers-who-never-order/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata) 
**Difficulty:** Easy  
**Date Solved:** 24-04-2025
# Solution:
**return customers[~customers.id.isin(orders.customerId.values)].loc[:,'name'].to_frame().rename(columns={"name": "Customers"})**