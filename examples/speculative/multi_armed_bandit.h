#ifndef MULTI_ARMED_BANDIT_EPSILON_GREEDY_UCB_H
#define MULTI_ARMED_BANDIT_EPSILON_GREEDY_UCB_H

#include <vector>
#include <tuple>
#include <limits>

class MultiArmedBanditEpsilonGreedyUCB {
public:
    MultiArmedBanditEpsilonGreedyUCB(int arms, double epsilon = 0.1, double c = 2);

    void pull_arm(int arm, double reward = std::numeric_limits<double>::quiet_NaN());
    int choose_arm();
    int best_arm();
    std::tuple<int, std::vector<double>, int> run(int total_pulls, double stop_threshold = 0.03, int min_pulls = 10);

private:
    int arms;
    double epsilon;
    double c;
    std::vector<double> estimated_means;
    std::vector<int> counts;
};

#endif // MULTI_ARMED_BANDIT_EPSILON_GREEDY_UCB_H