import numpy as np
import matplotlib.pyplot as plt

np.random.seed(200)

class LQR:
    def __init__(self):
        self.A = np.array([[1, .1],
                           [0, 1]])
        self.B = np.array([0, .1]).reshape(-1, 1)
        self.b = np.array([5, 0]).reshape(-1, 1)
        self.Sigma = np.diag([0.01, 0.01])
        self.K = np.array([5, 0.3])
        self.T = 50
        self.k = 0.3
        self.H = 1
        self.R = [np.diag([0.01, 0.1]), np.diag([1e5, 0.1])]
        self.r = [np.array([10, 0]).reshape(-1, 1),
                  np.array([20, 0]).reshape(-1, 1)]

    def gaussian(self, mean, cor, size=(2, 1)):
        """
        Contribute a multivariate Gausian distribution array
        :param mean:    [n, 1] array, Mean Array
        :param cor:     [n, n] array, Covariance Matrix
        :param size:    tuple
        :return G:      [n, 1] array, multivariate Gausian Matrix
        """
        if mean.shape != size:
            size = mean.shape

        G = np.zeros(size)
        for i in range(size[0]):
            G[i] = np.random.normal(mean[i], np.sqrt(cor[i, i]))
        return G

    def iteration(self, case="default"):
        """
        execute the iteration to compute the final states
        :param case:    string, case of different task
                                "default" : task 2.1
                                "P_Control" : task 2.2
                                "Optimal" : task 2.3
        :return s_list:  [2, n] array. all states under LQR
        :return a_list:  [1, n] array. all action under LQR
        :return rewards:  [1, n] array. rewards of each step
        """

        # initialize
        s_list = np.zeros((2, self.T + 1))
        a_list = np.zeros((1, self.T + 1))
        rewards = np.zeros((1, self.T + 1))

        if case == "Optimal":
            K_lst, k_lst = self.optimal()

        for i in range(0, self.T+1):
            # compute state
            if i == 0:
                s = self.gaussian(np.zeros((2, 1)), np.eye(2))
            else:
                s = self.A @ s + self.B * a + w

            if i < self.T:
                if case == "default":
                    a = -self.K @ s + self.k
                elif case == "P_Control":
                    a = self.p_controll(i, s)
                elif case == "Optimal":
                    a = K_lst[i] @ s + k_lst[i]
            else:
                a = 0
            w = self.gaussian(self.b, self.Sigma)

            # compute reward
            if i == 14 or i == 40:
                R = self.R[1]
            else:
                R = self.R[0]

            if i <= 14:
                r = self.r[0]
            else:
                r = self.r[1]

            reward = -(s - r).T @ R @ (s - r) - a * self.H * a

            # assign in array
            s_list[:, i] = s.reshape(-1)
            a_list[:, i] = a
            rewards[:, i] = reward

        return s_list, a_list, rewards

    def visualisation(self, execution=20, case="default"):
        """
        plot the mean and 95% confidence with 20 times execution
        :param execution:    int, execution times
        :param case:    string, case of different task
                                "default" : task 2.1
                                "P_Control" : task 2.2
                                "Optimal" : task 2.3
        """
        def statistic_proc(states_1, states_2, actions):
            """
            calculate the mean and std of each step over all executions
            :param states_1:    [m, n] array, first state over all executions
            :param states_2:    [m, n] array, second state over all executions
            :param actions:     [m, n] array, control action over all executions
            :return mean_s:     [2, n] array, mean of both states in each step
            :return std_s:      [2, n] array, std of both states in each step
            :return mean_a:     [1, n] array, mean of control action in each step
            :return std_a:      [1, n] array, mean of control action in each step
            """
            mean_s = np.zeros((2, self.T + 1))
            std_s = np.zeros((2, self.T + 1))
            mean_s[0, :] = np.mean(states_1, axis=0)
            mean_s[1, :] = np.mean(states_2, axis=0)
            std_s[0, :] = np.std(states_1, axis=0)
            std_s[1, :] = np.std(states_2, axis=0)
            mean_a = np.mean(actions, axis=0)
            std_a = np.std(actions, axis=0)

            return mean_s, std_s, mean_a, std_a

        # initial
        states_1 = np.zeros((execution, self.T + 1))
        states_2 = np.zeros((execution, self.T + 1))
        actions = np.zeros((execution, self.T + 1))
        rewards_cum = np.zeros((execution, 1))

        for i in range(execution):
            s_list, a_list, rewards = self.iteration(case)
            states_1[i, :] = s_list[0, :]
            states_2[i, :] = s_list[1, :]
            actions[i, :] = a_list.reshape(-1)
            rewards_cum[i, 0] = np.sum(rewards)

        print("cumulatgive reward: {}  +- {}".format(np.mean(rewards_cum), np.std(rewards_cum)))

        mean_s, std_s, mean_a, std_a = statistic_proc(states_1, states_2, actions)
        time_series = np.linspace(0, self.T, self.T+1)

        if case == "default":
            plt.figure()
            ax1 = plt.subplot(1, 3, 1)
            ax1.plot(time_series, mean_s[0, :], color='tab:blue')
            ax1.fill_between(time_series, 2 * std_s[0, :] + mean_s[0, :],
                             -2 * std_s[0, :] + mean_s[0, :], color='red', alpha=0.3)
            plt.xlabel('time in s')
            plt.title('first state')
            plt.grid(True)

            ax1 = plt.subplot(1, 3, 2)
            ax1.plot(time_series, mean_s[1, :], color='tab:blue')
            ax1.fill_between(time_series, 2 * std_s[1, :] + mean_s[1, :],
                             -2 * std_s[1, :] + mean_s[1, :], color='red', alpha=0.3)
            plt.xlabel('time in s')
            plt.title('second state')
            plt.grid(True)

            ax1 = plt.subplot(1, 3, 3)
            ax1.plot(time_series[:-1], mean_a[:-1], color='tab:blue')
            ax1.fill_between(time_series[:-1], 2 * std_a[:-1] + mean_a[:-1],
                             -2 * std_a[:-1] + mean_a[:-1], color='red', alpha=0.3)
            plt.xlabel('time in s')
            plt.title('control action')
            plt.grid(True)

        elif case == "P_Control":
            for i in range(execution):
                s_list, a_list, rewards = self.iteration("default")
                states_1[i, :] = s_list[0, :]
            mean_s[1, :] = np.mean(states_1, axis=0)
            std_s[1, :] = np.std(states_1, axis=0)

            plt.figure()
            line1, = plt.plot(time_series, mean_s[1, :], color='tab:blue')
            plt.fill_between(time_series, 2 * std_s[1, :] + mean_s[1, :],
                             -2 * std_s[1, :] + mean_s[1, :], color='blue', alpha=0.3)
            line2, = plt.plot(time_series, mean_s[0, :], color='tab:green')
            plt.fill_between(time_series, 2 * std_s[0, :] + mean_s[0, :],
                             -2 * std_s[0, :] + mean_s[0, :], color='green', alpha=0.3)
            plt.legend([line1, line2], ["desired s == 0", "desired s == r_t"])
            plt.title("first state under simple P Control")
            plt.grid(True)

        elif case == "Optimal":
            for i in range(execution):
                s_list, a_list, rewards = self.iteration("default")
                states_1[i, :] = s_list[0, :]
                states_2[i, :] = s_list[1, :]
                actions[i, :] = a_list.reshape(-1)
            mean_s_def, std_s_def, mean_a_def, std_a_def = \
                statistic_proc(states_1, states_2, actions)

            for i in range(execution):
                s_list, a_list, rewards = self.iteration("P_Control")
                states_1[i, :] = s_list[0, :]
                states_2[i, :] = s_list[1, :]
                actions[i, :] = a_list.reshape(-1)
            mean_s_p, std_s_p, mean_a_p, std_a_p = \
                statistic_proc(states_1, states_2, actions)

            namespace = ["comparing in first state", "comparing in second state"]
            for i in range(2):
                plt.figure()
                line1, = plt.plot(time_series, mean_s_def[i, :], color='tab:blue')
                plt.fill_between(time_series, 2 * std_s_def[i, :] + mean_s_def[i, :],
                                 -2 * std_s_def[i, :] + mean_s_def[i, :], color='blue', alpha=0.3)
                line2, = plt.plot(time_series, mean_s_p[i, :], color='tab:green')
                plt.fill_between(time_series, 2 * std_s_p[i, :] + mean_s_p[i, :],
                                 -2 * std_s_p[i, :] + mean_s_p[i, :], color='green', alpha=0.3)
                line3, = plt.plot(time_series, mean_s[i, :], color='red')
                plt.fill_between(time_series, 2 * std_s[i, :] + mean_s[i, :],
                                 -2 * std_s[i, :] + mean_s[i, :], color='red', alpha=0.3)
                plt.legend([line1, line2, line3], ["default", "simple P Control", "Optimal Control"])
                plt.grid(True)
                plt.title(namespace[i])

        plt.show()

    def p_controll(self, i, s):
        """
        modify control task as a simple P Controller
        :param i: int, iteration times
        :param s: int, actual state in i.th iteration
        :param k: int, constant action
        :return: a int, control action under P-Controller
        """
        if i <= 14:
            s_des = self.r[0]
        else:
            s_des = self.r[1]
        a = self.K @ (s_des - s) + self.k
        return a

    def optimal(self):
        """
        implement the Optimal LQR with reverse
        :return K_list:  [n x 1] list, catch of all actual Control Matrix
        :return k_list:  [n x 1] list, catch of all actual Control Constant
        """
        K_list = []
        k_list = []

        for i in reversed(range(1, self.T+1)):
            if i == 14 or i == 40:
                R = self.R[1]
            else:
                R = self.R[0]

            if i <= 14:
                r = self.r[0]
            else:
                r = self.r[1]

            if i == self.T:
                v_t = R @ r
                V_t = R
            else:
                v_t = R @ r + (self.A - M_t).T @ (v_t - V_t @ self.b)
                V_t = R + (self.A - M_t).T @ V_t @ self.A

            M_t = 1/(self.H + self.B.T @ V_t @ self.B) * self.B @ self.B.T @ V_t @ self.A  # V_t from last iteration

            K_t = - 1/(self.H + self.B.T @ V_t @ self.B) * self.B.T @ V_t @ self.A
            k_t = - 1/(self.H + self.B.T @ V_t @ self.B) * self.B.T @ (V_t @ self.b - v_t)

            # assignment
            K_list.insert(0, K_t)
            k_list.insert(0, k_t)

        return K_list, k_list


if __name__ == '__main__':
    tasks = ["default", "P_Control", "Optimal"]

    HW2 = LQR()
    # task 2.1
    HW2.visualisation(case=tasks[0])

    # task 2.2
    HW2.visualisation(case=tasks[1])

    # task 2.3
    HW2.visualisation(case=tasks[2])