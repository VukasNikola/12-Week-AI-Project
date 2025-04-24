# [2. Recyclable and Low Fat Products](https://leetcode.com/problems/recyclable-and-low-fat-products/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata) 
**Difficulty:** Easy  
**Date Solved:** 24-04-2025
# Solution:
**return (products[(products.low_fats == 'Y') & (products.recyclable == 'Y')].iloc[:,0]).to_frame()**