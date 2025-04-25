# [4. Artivle Views I](https://leetcode.com/problems/article-views-i/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata) 
**Difficulty:** Easy  
**Date Solved:** 25-04-2025
# Solution:
return(
        views
        .loc[views.author_id == views.viewer_id, 'author_id']
        .to_frame()
        .drop_duplicates()
        .rename(columns={'author_id':'id'})
        .sort_values(by='id'))