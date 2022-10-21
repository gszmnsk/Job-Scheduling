from JobShopEnv import JobShopEnv
from DQNAgent import DQNAgent

number_of_jobs = 4
number_of_machines = 3
number_of_features = 2
state_size = number_of_jobs * number_of_features
action_size = number_of_jobs
agent = DQNAgent(state_size, action_size, number_of_jobs, number_of_features)

# The loop for each job shop problem
for episode in range(EPISODES):
    problem = JobShopEnv(4, 3, 2)
    action = None
    current_state, score, done = problem.make_step()
    action_list = []
    old_score = 0
    number_of_success = 0

    # Loop for each step in range of 4*3
    for step in range(number_of_jobs * number_of_machines):
        action = agent.act(current_state)
        next_state, score, done = problem.make_step(action)
        if not done:
            reward = old_score - score + 15  # the hardest part of entire RL problem - finding decent reward function
        else:
            reward = -1000
        old_score = score
        agent.update_memory(current_state, action, reward, next_state, done)
        current_state = next_state

        if done:
            # if it reaches the 11 or 12 step, call it a success, as it processed all the tasks
            if step >= number_of_jobs * number_of_machines-1:
                number_of_success += 1
            break

        action_list.append(action)

        # train only if there are samples in a minibatch
        if len(agent.memory) > MINIBATCH_SIZE:
            agent.train(done)

        # for every 10 episodes print some stats
        if episode % 10 == 0:
            print(f"episode: {episode}/{EPISODES}, score: {score}, number of success: {number_of_success}, epsilon: {agent.epsilon}")
            print(action_list, len(action_list))
            print(problem.print_information())