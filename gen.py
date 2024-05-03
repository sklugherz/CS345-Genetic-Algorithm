from data import *
from agent import *
from input_vector import *


class gen:

    def gen(generations, population_size):
        ticker_data = data()
        generation_size = data.get_size // 4
        initial_population = []

        for _ in range(generations):
            #  create initial population
            if len(initial_population) == 0:
                for _ in range(population_size):
                    new_agent = agent()
                    initial_population.append(new_agent)
            
            #  one generation of agents running on the data
            for _ in range(generation_size):
                for agent in initial_population:
                    agent.update_trades()
                    agent.input(data.get_tensor())
                    # agent.run_brain()
                    # agent.make_trades()
                ticker_data.next()

            #  rank agents
            ranked_agents = []

            for agent in initial_population:
                ranked_agents.append( (agent.get_fitness),agent )

            ranked_agents.sort()
            ranked_agents.reverse()

            #  choose elites
            elites = []
            for e in range(ranked_agents // 2):
                elites.append(ranked_agents[e][1])

            #  get weight elements
            elements = []
            for elite in elites:
                e1 = elite.get_weight_vector()[:len(elite.get_weight_vector()) // 2]
                e2 = elite.get_weight_vector()[len(elite.get_weight_vector()) // 2:]

                elements.append((e1, e2))

            #  create children
            new_generation = []
            for _ in range(population_size // 2):
                parent1 = random.choice(elements)
                parent2 = random.choice(elements)

                e1 = parent1[0]
                e2 = parent1[1]
                e3 = parent2[0]
                e4 = parent2[1]

                weight1 = e1 + e4
                weight2 = e3 + e2

                new_generation.append(agent(weight1))
                new_generation.append(agent(weight2))

            initial_population = new_generation
