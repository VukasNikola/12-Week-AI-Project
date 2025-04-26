# [7. Fix Names in a Table](https://leetcode.com/problems/fix-names-in-a-table/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata) 
**Difficulty:** Easy  
**Date Solved:** 26-04-2025
# Solution:
    users.name = users.loc[:,'name'].str.capitalize()
    return users.sort_values(by='user_id', ascending= True)