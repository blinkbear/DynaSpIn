#include "multi_armed_bandit.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <chrono>
#include <algorithm>
#include <tuple>

MultiArmedBanditEpsilonGreedyUCB::MultiArmedBanditEpsilonGreedyUCB(int arms, double epsilon, double c)
    : arms(arms), epsilon(epsilon), c(c), estimated_means(arms, 0.0), counts(arms, 0) {}

void MultiArmedBanditEpsilonGreedyUCB::pull_arm(int arm, double reward) {
    if (std::isnan(reward)) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(estimated_means[arm], 1);
        reward = d(gen);
    }
    counts[arm] += 1;
    estimated_means[arm] = (estimated_means[arm] * (counts[arm] - 1) + reward) / counts[arm];
    // std::cout << "arm: " << arm << ", reward: " << reward << ", estimated mean: " << estimated_means[arm] << std::endl;
}

int MultiArmedBanditEpsilonGreedyUCB::choose_arm() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    if (dis(gen) < epsilon) {
        std::uniform_int_distribution<> int_dis(0, arms - 1);
        return int_dis(gen);
    } else {
        std::vector<double> ucb(arms);
        double total_counts = std::accumulate(counts.begin(), counts.end(), 0.0);
        for (int i = 0; i < arms; ++i) {
            ucb[i] = estimated_means[i] + c * std::sqrt(2 * std::log(total_counts) / counts[i]);
        }
        return std::distance(ucb.begin(), std::max_element(ucb.begin(), ucb.end()));
    }
}

int MultiArmedBanditEpsilonGreedyUCB::best_arm() {
    return std::distance(estimated_means.begin(), std::max_element(estimated_means.begin(), estimated_means.end()));
}

std::tuple<int, std::vector<double>, int> MultiArmedBanditEpsilonGreedyUCB::run(int total_pulls, double stop_threshold, int min_pulls) {
    std::vector<double> total_rewards(total_pulls, 0.0);
    double best_arm_estimate = estimated_means[best_arm()] == 0 ? 0.1 : estimated_means[best_arm()];
    bool converged = false;
    int patience = 0;
    int final_round = total_pulls;

    for (int i = 0; i < total_pulls; ++i) {
        int arm;
        double reward;
        if (i < arms) {
            arm = i;
            reward = 0.1;
            counts[arm] += 1;
        } else {
            arm = choose_arm();
            pull_arm(arm);
            reward = estimated_means[arm];
        }
        total_rewards[i] = reward;

        if (i >= min_pulls && !converged) {
            int current_best_arm = best_arm();
            double current_best_estimate = estimated_means[current_best_arm];
            // std::cout << "ratio: " << std::abs(current_best_estimate - best_arm_estimate) / best_arm_estimate << std::endl;
            if (std::abs(current_best_estimate - best_arm_estimate) / best_arm_estimate < stop_threshold) {
                patience += 1;
                if (patience >= 5) {
                    converged = true;
                    // std::cout << "Converged at round " << i << " with best arm: " << current_best_arm
                            //   << ", estimated mean: " << current_best_estimate
                            //   << ", best_arm_estimate is " << best_arm_estimate << std::endl;
                    final_round = i;          
                    break;
                }
            }
            best_arm_estimate = current_best_estimate;
        }
    }

    if (!converged) {
        std::cout << "Did not converge after " << final_round << " pulls." << std::endl;
    }

    return {best_arm(), total_rewards, final_round};
}