import random
from collections import defaultdict

import matplotlib.pyplot as plt

# =========================
# Core Simulation Constants
# =========================
TARGET_CITIZENS = 30000  # Target population to reach
YEARS = 250              # Number of years to simulate
POPULATION_SIZE = 100    # Number of strategies in GA population
GENERATIONS = 20         # Number of GA generations
MUTATION_RATE = 0.1      # Probability of mutation
BASE_GROWTH_RATE = 0.01  # Base population growth rate (1% per year)

# =====================
# Scenario Definitions
# =====================
SCENARIOS = {
    "default": {
        "drought_chance": 0.05,
        "invasion_chance": 0.03,
        "plague_chance": 0.03,
        "food_consumption": 0.08,
        "growth_modifier": 1.0
    },
    "hard_mode": {
        "drought_chance": 0.15,
        "invasion_chance": 0.08,
        "plague_chance": 0.08,
        "food_consumption": 0.08,
        "growth_modifier": 1.0
    },
    "peaceful": {
        "drought_chance": 0.01,
        "invasion_chance": 0.005,
        "plague_chance": 0.005,
        "food_consumption": 0.08,
        "growth_modifier": 1.2
    }
}

# =====================
# Core Functions
# =====================
def create_individual():
    food = random.random()
    military = random.random() * (1 - food)
    medicine = 1 - food - military
    return [food, military, medicine]

def simulate_game(strategy, events, scenario_params, track_history=False):
    initial_citizens = 100.0
    citizens = initial_citizens
    food_stock = 100
    military_score = 50
    medicine_stock = 50
    penalties = 0

     # History tracking variables
    if track_history:
        citizens_history = []
        food_history = []
        military_history = []
        medicine_history = []
        event_years = {'drought': [], 'invasion': [], 'plague': []}
    
    for year in range(YEARS):
        # Normalize strategy
        total = sum(strategy)
        food_allocation = strategy[0] / total
        military_allocation = strategy[1] / total
        medicine_allocation = strategy[2] / total

        # Resource production with scenario modifier
        food_production = food_allocation * citizens * 0.1 * scenario_params["growth_modifier"]
        military_investment = military_allocation * citizens * 0.05
        medicine_investment = medicine_allocation * citizens * 0.03

        # Consumption with scenario modifier
        food_consumption = citizens * scenario_params["food_consumption"]
        food_stock += food_production - food_consumption

        # Handle starvation
        if food_stock <= 0:
            citizens *= 0.7
            food_stock = 0
            penalties += 30
            if citizens < 0:
                citizens = 0

        military_score += military_investment
        medicine_stock += medicine_investment

         # Process events
        event = events[year]
        if event == "drought":
            food_stock -= 20
            if track_history:
                event_years['drought'].append(year)
        elif event == "invasion":
            if military_score < 30:
                citizens *= 0.8
                food_stock *= 0.8
                medicine_stock *= 0.8
                penalties += 30
            if track_history:
                event_years['invasion'].append(year)
        elif event == "plague":
            citizens *= 0.7
            penalties += 30
            if track_history:
                event_years['plague'].append(year)

        if citizens <= 0:
            citizens = 0
            if track_history:
                citizens_history.append(citizens)
                food_history.append(food_stock)
                military_history.append(military_score)
                medicine_history.append(medicine_stock)
            break

        # Population growth
        growth_rate = BASE_GROWTH_RATE * scenario_params["growth_modifier"]
        if food_stock > 0:
            growth_rate += 0.02
        if medicine_stock > 50:
            growth_rate += 0.01
        citizens += citizens * growth_rate

        
        if track_history:
            citizens_history.append(citizens)
            food_history.append(food_stock)
            military_history.append(military_score)
            medicine_history.append(medicine_stock)
        

    population_increase = citizens - initial_citizens
    fitness = population_increase - penalties
    if citizens >= TARGET_CITIZENS:
        fitness += 1000
    
    if track_history:
        return (fitness, int(citizens), citizens_history, food_history, military_history, medicine_history, event_years)
    else:
        return fitness, int(citizens), food_stock, military_score, medicine_stock

def generate_events(scenario_params):
    events = []
    for _ in range(YEARS):
        if random.random() < scenario_params["drought_chance"]:
            events.append("drought")
        elif random.random() < scenario_params["invasion_chance"]:
            events.append("invasion")
        elif random.random() < scenario_params["plague_chance"]:
            events.append("plague")
        else:
            events.append(None)
    return events

def tournament_selection(population, fitness_scores, tournament_size=3):
    contestants = random.sample(list(zip(population, fitness_scores)), tournament_size)
    return max(contestants, key=lambda x: x[1][0])[0]

def evolve_population(population, events, scenario_params):
    fitness_scores = [simulate_game(individual, events, scenario_params) for individual in population]
    new_population = []
    while len(new_population) < POPULATION_SIZE:
        parent1 = tournament_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores)
        child = [(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)]
        if random.random() < MUTATION_RATE:
            child[random.randint(0, 2)] = random.random()
        total = sum(child)
        child = [x / total for x in child]
        new_population.append(child)
    return new_population

# =====================
# Testing Framework
# =====================
def test_strategy(strategy, num_tests=10):
    results = defaultdict(list)
    
    for scenario_name, params in SCENARIOS.items():
        for _ in range(num_tests):
            events = generate_events(params)
            fitness, citizens, _, _, _ = simulate_game(strategy, events, params)
            results[scenario_name].append({
                "fitness": fitness,
                "citizens": citizens,
                "success": citizens >= TARGET_CITIZENS
            })
    
    # Print summary
    print("\nStrategy Performance Across Scenarios:")
    for scenario, data in results.items():
        success_rate = sum(1 for r in data if r["success"]) / num_tests
        avg_citizens = sum(r["citizens"] for r in data) / num_tests
        print(f"{scenario.upper():<10} | Success: {success_rate*100:.0f}% | Avg Pop: {avg_citizens:.0f}")


# =====================
# Graphics 
# =====================
def plot_strategy_allocation(strategy):
    labels = ['Food', 'Military', 'Medicine']
    sizes = [x/sum(strategy) for x in strategy]  # Ensure normalized
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
            colors=['#4CAF50', '#F44336', '#2196F3'],
            startangle=90, explode=(0.05, 0, 0))
    plt.title('Optimal Resource Allocation Strategy', pad=20)
    plt.tight_layout()
    plt.show()

def plot_population_growth(strategy, scenario_name="default"):
    scenario = SCENARIOS[scenario_name]
    events = generate_events(scenario)
    result = simulate_game(strategy, events, scenario, track_history=True)
    
    _, _, citizens, food, military, medicine, event_years = result
    
    plt.figure(figsize=(12, 6))
    
    # Main population line
    plt.plot(citizens, label='Population', linewidth=3, color='#9C27B0')
    
    # Event markers
    for event, years in event_years.items():
        if years:
            y_values = [citizens[year] for year in years]
            color = {
                'drought': '#FF9800',
                'invasion': '#F44336',
                'plague': '#607D8B'
            }[event]
            plt.scatter(years, y_values, label=event.capitalize(), 
                       color=color, s=100, zorder=3)
    
    plt.axhline(y=TARGET_CITIZENS, color='#4CAF50', linestyle='--', 
                label='Target Population')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.title(f'Population Growth ({scenario_name.capitalize()} Scenario)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_resource_levels(strategy, scenario_name="default"):
    scenario = SCENARIOS[scenario_name]
    events = generate_events(scenario)
    result = simulate_game(strategy, events, scenario, track_history=True)
    
    _, _, citizens, food, military, medicine, _ = result
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(food, label='Food Stock', color='#4CAF50')
    plt.plot(military, label='Military Score', color='#F44336')
    plt.plot(medicine, label='Medicine Stock', color='#2196F3')
    
    plt.xlabel('Year')
    plt.ylabel('Resource Amount')
    plt.title(f'Resource Levels Over Time ({scenario_name.capitalize()} Scenario)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

    event_years = {
        'drought': [],
        'invasion': [],
        'plague': []
    }
    for year, event in enumerate(events):
        if event in event_years:
            event_years[event].append(year)

    plt.figure(figsize=(12, 3))
    for event, years in event_years.items():
        plt.scatter(years, [1]*len(years), label=event, s=100)
    plt.yticks([])
    plt.xlabel('Year')
    plt.title('Event Timeline')
    plt.legend()
    plt.show()

def plot_scenario_comparison(strategy, num_tests=10):
    results = {}
    
    for scenario_name, params in SCENARIOS.items():
        successes = 0
        avg_pop = 0
        for _ in range(num_tests):
            events = generate_events(params)
            fitness, citizens, *_ = simulate_game(strategy, events, params)
            if citizens >= TARGET_CITIZENS:
                successes += 1
            avg_pop += citizens
        results[scenario_name] = {
            'success_rate': successes / num_tests,
            'avg_population': avg_pop / num_tests
        }
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Success rate
    names = list(results.keys())
    success_rates = [x['success_rate'] for x in results.values()]
    colors = ['#4CAF50', '#F44336', '#2196F3']
    
    ax1.bar(names, success_rates, color=colors)
    ax1.set_title('Success Rate (%)')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(success_rates):
        ax1.text(i, v + 0.05, f"{v:.0%}", ha='center')
    
    # Average population
    avg_pops = [x['avg_population'] for x in results.values()]
    ax2.bar(names, avg_pops, color=colors)
    ax2.set_title('Average Final Population')
    ax2.axhline(TARGET_CITIZENS, color='red', linestyle='--')
    for i, v in enumerate(avg_pops):
        ax2.text(i, v + 1000, f"{int(v):,}", ha='center')
    
    fig.suptitle('Strategy Performance Across Scenarios', y=1.05)
    plt.tight_layout()
    plt.show()
    plt.show()


    plt.figure(figsize=(8, 5))
    plt.plot(fitness_history)
    plt.title('Genetic Algorithm Progress')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.grid(True)
    plt.show()   
# =====================
# Main Execution
# =====================
if __name__ == "__main__":
    # Training phase (using default scenario)
    print("Training optimal strategy...")
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    for generation in range(GENERATIONS):
        events = generate_events(SCENARIOS["default"])
        population = evolve_population(population, events, SCENARIOS["default"])
        
        # Track best performer
        best_fitness, best_citizens, _, _, _ = max(
            [simulate_game(ind, events, SCENARIOS["default"]) for ind in population],
            key=lambda x: x[0]
        )
        print(f"Gen {generation}: Fitness={best_fitness:.0f}, Pop={best_citizens}")

    # Identify best strategy
    best_strategy = max(population, 
                       key=lambda x: simulate_game(x, generate_events(SCENARIOS["default"]), 
                                                 SCENARIOS["default"])[0])
    print("\nFinal Strategy:", [f"{x:.2f}" for x in best_strategy])

    # Cross-test the strategy
    test_strategy(best_strategy)

     # After finding best_strategy:
    print("\nVisualizing results...")
    plot_strategy_allocation(best_strategy)
    
    # Show for default scenario
    plot_population_growth(best_strategy, "default")
    plot_resource_levels(best_strategy, "default")
    
    # Show for hard mode
    plot_population_growth(best_strategy, "hard_mode")
    
    # Scenario comparison
    plot_scenario_comparison(best_strategy)