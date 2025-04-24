# [1. Big Countries](https://leetcode.com/problems/big-countries/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata)
**Difficulty:** Easy  
**Date Solved:** 23-04-2025
# Solution:
**return world[(world['area'] >= 3000000) | (world['population'] >= 25000000)].loc[:,['name','area','population']]**