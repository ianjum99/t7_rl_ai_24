#include "replay_buffer.h"

ReplayBuffer::ReplayBuffer(int capacity) : capacity(capacity), position(0) {
    buffer.reserve(capacity);
}

void ReplayBuffer::add(std::vector<float> state, int action, float reward, std::vector<float> next_state, bool done) {
    if (buffer.size() < capacity) {
        buffer.push_back({state, action, reward, next_state, done});
    } else {
        buffer[position] = {state, action, reward, next_state, done};
    }
    position = (position + 1) % capacity;
}

std::tuple<std::vector<float>, int, float, std::vector<float>, bool> ReplayBuffer::sample(int index) {
    return buffer[index];
}

