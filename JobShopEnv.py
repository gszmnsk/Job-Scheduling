import numpy as np


class JobShopEnv:
    """
    The enviroment for Job Shop problem
    """

    # One job has multiple tasks that have to be processed in specific order and have different processing time

    # matrix M_processing_order[i,j] the machine processing order of the tasks of each job
    # i - job, j - number of machine for the task
    M_processing_order = np.array([[1, 0, 2], [0, 2, 1], [1, 2, 0], [1, 0, 2]])

    # matrix M_processing_time[i,j] with the time needed for every task in a job
    # i - job, j - time needed for the task
    M_processing_time = np.array([[18, 20, 17], [18, 15, 16], [17, 18, 23], [18, 21, 15]])

    number_of_jobs = None
    number_of_machines = None
    number_of_features = None

    M_start_time = None
    M_end_time = None
    M_schedule_plan = None
    schedule_line = None

    def __init__(self, number_of_jobs, number_of_machines, number_of_features):
        self.number_of_jobs = number_of_jobs
        self.number_of_machines = number_of_machines
        self.schedule_line = []  # [[action_1, job_position_1], ...]
                                 # action - an index of considered job, job_position - an index of a task to do
        self.number_features = number_of_features

    def generate_possible_job_position(self):
        """
        Method generates possible job positions
        :return job_index_list, job_position_list:
        """
        job_position_list = [0 for i in range(self.number_of_jobs)]
        for job_id, job_position in self.schedule_line:
            if job_position < self.number_of_machines - 1:  # (counting from zero) if the job process is not finished
                job_position_list[job_id] = job_position + 1    # counting the number of finished tasks in that schedule line,
                                                                # generate the next set of tasks to do in each job
            else:
                job_position_list[job_id] = -1  # if it was the last task, the job process is finished
        job_index_list = [[i, job_position_list[i]] for i in range(self.number_of_jobs)]
        return job_index_list, job_position_list

    def evaluate_action(self, schedule_line):
        """
        Returns the total time of processed jobs in a schedule, from start to finish
        The output of the method is used for the score and for further reward calculation
        :param schedule_line:
        :return makespan:
        """
        M_start_time = np.zeros((self.number_of_machines, self.number_of_jobs))
        M_end_time = np.zeros((self.number_of_machines, self.number_of_jobs))

        machine_timeline = np.zeros((self.number_of_machines), dtype=int)
        machine_index = np.zeros((self.number_of_machines), dtype=int)

        job_timeline = np.zeros((self.number_of_jobs), dtype=int)
        job_index = np.zeros((self.number_of_jobs), dtype=int)

        M_schedule_plan = np.zeros((self.number_of_machines, self.number_of_jobs, 2), dtype=int)

        for job_id, job_position in schedule_line:
            machine_id = self.M_processing_order[job_id, job_position]
            task_time = self.M_processing_time[job_id, job_position]

            current_start_time = max(machine_timeline[machine_id], job_timeline[job_id])
            current_end_time = current_start_time + task_time

            # update the timeline of each machine and each job
            machine_timeline[machine_id], job_timeline[job_id] = current_end_time, current_end_time

            # the index of the job order on a certain machine
            current_index = machine_index[machine_id]
            M_start_time[machine_id, current_index] = current_start_time
            M_end_time[machine_id, current_index] = current_end_time

            # making a schedule by adding [job_id, job_position] by order to each machine queue
            M_schedule_plan[machine_id, current_index, :] = [job_id, job_position]

            machine_index[machine_id] += 1  # next machine
            job_index[job_id] += 1  # next job

        self.M_start_time = M_start_time
        self.M_end_time = M_end_time
        self.M_schedule_plan = M_schedule_plan
        makespan = np.max(M_end_time)

        return makespan


    def get_features_of_one_job(self,  job_id, job_position):
        """
        Returns single task of a job features that are going to be fed into get_features_of_current_state method
        :param job_position, job_id:
        :return job_features:
        """
        machine_id = self.M_processing_order[job_id, job_position]
        job_end_time = np.sum(self.M_processing_time[job_id, :job_position])
        job_time_need = np.sum(self.M_processing_time[job_id, :])  # the total time needed to complete each job
        machine_end_time = np.max(self.M_end_time, axis=1)  # the total time of all completed jobs on particular machine

        job_features = []
        if job_position == 0:
            current_machine_end_time = 0.5
            current_job_end_time = 0.5

        else:
            # when a particular task was running on a machine, what was the end time on that machine
            current_machine_end_time = machine_end_time[machine_id] / (np.average(machine_end_time))

            current_job_end_time = job_end_time / job_time_need

        job_features.append(current_machine_end_time)
        job_features.append(current_job_end_time)

        self.number_of_features = len(job_features)

        # when the task is the last one:
        if job_position == -1:
            job_features = [-1] * self.number_features

        return job_features

    def get_features_of_current_state(self, possible_job_position):
        """
        Returns features that the state is made of
        :param possible_job_position:
        :return state_features:
        """
        state_features = []
        for job_id, job_position in possible_job_position:
            job_feature = self.get_features_of_one_job(job_id, job_position)
            state_features.append(job_feature)
        return state_features

    def make_step(self, action=None):
        """
        Method for making steps
        :param action:
        :return state, score, done:
        """

        done = False        # the step is not done
        if action == None:  # if there is no action (which is an initial state), generate some state
            self.evaluate_action(self.schedule_line)  # get the M_end_time matrix
            possible_job_position = self.generate_possible_job_position()[0]
            state = np.array(self.get_features_of_current_state(possible_job_position))
            score = 0
        else:
            possible_job_position_generator = self.generate_possible_job_position()
            job_position_list = possible_job_position_generator[1]
            if job_position_list[action] == -1:  # if the task was the last one
                done = True
                actions_to_choose = [[i, job_position_list[i]] for i in range(self.number_of_jobs)
                                     if job_position_list[i] != -1]   # choose the action that is not already done
                chosen_action = actions_to_choose[0]  # choose another task from another job
            else:   # if the job is not done
                chosen_action = [action, job_position_list[action]]

            self.schedule_line.append(chosen_action)    # making a schedule line of possible actions to take

            score = self.evaluate_action(self.schedule_line)  # evaluating the schedule of the actions

            possible_job_position = possible_job_position_generator[0]
            state = np.array(self.get_features_of_current_state(possible_job_position))

        # putting state into a correct shape for the neural network that agent uses
        state = [np.reshape(state[i], (1,2)) for i in range(self.number_of_jobs)]

        return state, score, done

    def print_information(self):

        print('Processing order: ', self.M_processing_order)
        print('Processing time: ', self.M_processing_time)
        print('Start time: ', self.M_start_time)
        print('End time: ', self.M_end_time)
        print('The task schedule: ', self.M_schedule_plan)



#test
problem = JobShopEnv(4, 3, 2)
print(problem.evaluate_action([[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]]))
array = problem.make_step()[0]
print(array)
print(np.shape(array))





