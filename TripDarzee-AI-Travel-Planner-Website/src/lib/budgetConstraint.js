// utils/budgetConstraint.js
export const calculateFoodCost = (budgetLevel) => {
    switch (budgetLevel) {
      case "low":
        return 100 * 2;
      case "medium":
        return 300 * 2;
      case "high":
        return 800 * 2;
      default:
        throw new Error("Invalid budget level. Choose 'low', 'medium', or 'high'.");
    }
  };
  
  export const calculateActualBudget = (hotelCost, totalDistance, noOfDays, noOfPeople, budgetLevel) => {
    const foodCost = calculateFoodCost(budgetLevel);
    return (
      hotelCost +
      totalDistance * 20 +
      (1 / 10) * (hotelCost + totalDistance) +
      foodCost * noOfDays * noOfPeople
    );
  };
  
  export const calculateBudgetScore = (actualBudget, expectedBudget) => {
    return actualBudget / expectedBudget;
  };