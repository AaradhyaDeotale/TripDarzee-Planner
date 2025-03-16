def calculate_food_cost(budget_level):
    if budget_level == "low":
        return 100 * 2
    elif budget_level == "medium":
        return 300 * 2
    elif budget_level == "high":
        return 800 * 2
    else:
        raise ValueError("Invalid budget level. Choose 'low', 'medium', or 'high'.")

def calculate_actual_budget(hotel_cost, total_distance, no_of_days, no_of_people, budget_level):
    food_cost = calculate_food_cost(budget_level)
    actual_budget = (hotel_cost + total_distance * 20 + (1/10) * (hotel_cost + total_distance) + food_cost * no_of_days * no_of_people)
    return actual_budget

def calculate_budget_score(actual_budget, expected_budget):
    return actual_budget / expected_budget

# Example Usage
expected_budget = 5000  # Replace with actual expected budget
hotel_cost = 2000        # Replace with actual hotel cost
total_distance = 300     # Replace with actual distance in km
no_of_days = 5           # Replace with actual number of days
no_of_people = 3         # Replace with actual number of people
budget_level = "medium"  # Choose from "low", "medium", "high"

actual_budget = calculate_actual_budget(hotel_cost, total_distance, no_of_days, no_of_people, budget_level)
budget_score = calculate_budget_score(actual_budget, expected_budget)

print(f"Actual Budget: {actual_budget}")
print(f"Budget Score: {budget_score}")
