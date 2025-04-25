# [5. Invalid Tweets](https://leetcode.com/problems/invalid-tweets/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata) 
**Difficulty:** Easy  
**Date Solved:** 25-04-2025
# Solution:
return (
        tweets
        .query('content.str.len() > 15')
        .loc[:,'tweet_id']
        .to_frame())