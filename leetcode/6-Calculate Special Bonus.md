# [6. Calculate Special Bonus](https://leetcode.com/problems/calculate-special-bonus/description/?envType=study-plan-v2&envId=30-days-of-pandas&lang=pythondata) 
**Difficulty:** Easy  
**Date Solved:** 26-04-2025
# Solution:
    employees['bonus'] = 0
    employees.loc[(employees.employee_id%2 == 1) & ~(employees.name.str.contains("M")), "bonus"] = employees.salary

    return employees.loc[:,('employee_id', 'bonus')].sort_values(by='employee_id', ascending= True)