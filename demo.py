import random

# Constants
TARGET_CITIZENS = 10000  # Target population to reach
YEARS = 100              # Number of years to simulate (extended to 100)
POPULATION_SIZE = 100    # Number of strategies in the GA population
GENERATIONS = 20         # Number of GA generations
MUTATION_RATE = 0.1      # Probability of mutation
BASE_GROWTH_RATE = 0.01   # Base population growth rate (10% per year)

# Chromosome: [food_allocation, military_allocation, medicine_allocation]
def create_individual():
    # Generate random allocations that sum to 1.0
    food = random.random()
    military = random.random() * (1 - food)
    medicine = 1 - food - military
    return [food, military, medicine]

def simulate_game(strategy, events):
    # Initialize game state
    initial_citizens = 100.0  # Starting number of citizens (as a float)
    citizens = initial_citizens
    food_stock = 100     # Starting food stock
    military_score = 50  # Starting military score
    medicine_stock = 50  # Starting medicine stock

    # Track penalties for fitness calculation
    penalties = 0

    # Simulate years
    for year in range(YEARS):
        # Apply strategy (ensure allocations sum to 1.0)
        total = sum(strategy)
        food_allocation = strategy[0] / total
        military_allocation = strategy[1] / total
        medicine_allocation = strategy[2] / total

        # Scale investments with population (adjusted scaling factors)
        food_production = food_allocation * citizens * 0.1  # Food scaling
        military_investment = military_allocation * citizens * 0.05  # Military scaling
        medicine_investment = medicine_allocation * citizens * 0.03  # Medicine scaling

        # Update game state
        food_consumption = citizens * 0.08  # Reduced food consumption rate
        food_stock += food_production - food_consumption

        # Handle starvation if food stock is negative
        if food_stock <= 0:
            citizens *= 0.7  # Kill 20% of the population (reduced from 30%)
            food_stock = 0  # Reset food stock to zero
            penalties += 30  # Reduced fitness penalty for starvation
            if citizens < 0:
                citizens = 0  # Ensure citizens don't go below zero

        military_score += military_investment
        medicine_stock += medicine_investment

        # Apply pre-generated events for this year
        event = events[year]
        if event == "drought":
            food_stock -= 20  # Reduced impact of drought
        elif event == "invasion":
            if military_score < 30:
                citizens *= 0.8  # Lose 10% of citizens (reduced from 15%)
                food_stock *= 0.8  # Lose 15% of food (reduced from 20%)
                medicine_stock *= 0.8  # Lose 20% of medicine (reduced from 30%)
                penalties += 30  # Reduced fitness penalty for invasion
        elif event == "plague":
            citizens *= 0.7  # Kill 15% of the population (reduced from 20%)
            penalties += 30  # Reduced fitness penalty for plague

        # Check for game over conditions
        if citizens <= 0:
            break  # End simulation if no citizens left

        # Population growth (base growth rate + modifiers)
        growth_rate = BASE_GROWTH_RATE
        if food_stock > 0:  # Positive growth if there's enough food
            growth_rate += 0.02  # Slightly higher growth rate
        if medicine_stock > 50:  # Positive growth if medicine is abundant
            growth_rate += 0.01
        citizens += citizens * growth_rate  # Increase citizens (no int() truncation)

    # Fitness score based on population increase
    population_increase = citizens - initial_citizens
    fitness = population_increase - penalties  # Subtract penalties from fitness
    if citizens >= TARGET_CITIZENS:
        fitness += 1000  # Bonus for reaching the target
    return fitness, int(citizens), food_stock, military_score, medicine_stock  # Convert citizens to int for display

def generate_events():
    # Generate a list of events for each year
    events = []
    for _ in range(YEARS):
        if random.random() < 0.05:  # Drought (5% chance, reduced from 8%)
            events.append("drought")
        elif random.random() < 0.03:  # Invasion (3% chance, reduced from 4%)
            events.append("invasion")
        elif random.random() < 0.03:  # Plague (3% chance, reduced from 4%)
            events.append("plague")
        else:
            events.append(None)  # No event
    return events

def tournament_selection(population, fitness_scores, tournament_size=3):
    # Randomly select 'tournament_size' individuals
    contestants = random.sample(list(zip(population, fitness_scores)), tournament_size)
    # Return the individual with the highest fitness
    return max(contestants, key=lambda x: x[1][0])[0]


def evolve_population(population, events):
    # Evaluate fitness
    fitness_scores = [simulate_game(individual, events) for individual in population]

    # Crossover and mutation
    new_population = []
    while len(new_population) < POPULATION_SIZE:
       parent1 = tournament_selection(population, fitness_scores)
       parent2 = tournament_selection(population, fitness_scores)
       child = [(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)]  # Crossover
       if random.random() < MUTATION_RATE:
            child[random.randint(0, 2)] = random.random()  # Mutation
        # Normalize allocations
       total = sum(child)
       child = [x / total for x in child]
       new_population.append(child)
    return new_population



# Main loop
population = [create_individual() for _ in range(POPULATION_SIZE)]
for generation in range(GENERATIONS):
    # Generate events for this generation
    events = generate_events()
    
    # Evolve the population
    population = evolve_population(population, events)
    
    # Evaluate the best individual
    best_fitness, best_citizens, best_food, best_military, best_medicine = max([simulate_game(individual, events) for individual in population], key=lambda x: x[0])
    print(f"Generation {generation}:")
    print(f"  Best Fitness: {best_fitness}")
    print(f"  Citizens: {best_citizens}, Food: {best_food}, Military: {best_military}, Medicine: {best_medicine}")
    print()

# Best strategy
best_strategy = max(population, key=lambda x: simulate_game(x, generate_events())[0])
print("Best Strategy:", best_strategy)