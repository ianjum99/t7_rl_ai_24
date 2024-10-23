#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H

#include <vector>
#include <tuple>

// Replay buffer class for storing experiences
class ReplayBuffer {
private:
    std::vector<std::tuple<std::vector<float>, int, float, std::vector<float>, bool>> buffer;
    int capacity;
    int position;

public:
    // Constructor
    ReplayBuffer(int capacity);

    // Add experience to the buffer
    void add(std::vector<float> state, int action, float reward, std::vector<float> next_state, bool done);

    // Sample an experience from the buffer
    std::tuple<std::vector<float>, int, float, std::vector<float>, bool> sample(int index);
};

#endif  // REPLAY_BUFFER_H

