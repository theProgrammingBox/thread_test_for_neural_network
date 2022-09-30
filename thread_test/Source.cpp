#include <thread>
#include <iostream>
#include <chrono>

using std::thread;
using std::cout;
using std::endl;
using std::swap;

void initWeightsAndBiases(float* weights, float* biases, uint64_t numNodes, uint64_t row, uint16_t numRows) {
	for (uint64_t i = 0; i < numRows; i++) {
		for (uint64_t j = 0; j < numNodes; j++) {
			weights[(row + i) * numNodes + j] = (row + i) == j;
		}
		biases[row + i] = 0.0f;
	}
}

void multiplyWeightsAndBiases(float* weights, float* biases, float* inputs, float* outputs, float* perseptions, float* actions, uint64_t numNodes, uint64_t row, uint16_t numRows) {
	for (uint64_t i = 0; i < numRows; i++) {
		outputs[row + i] = biases[row + i];
		for (uint64_t j = 0; j < numNodes; j++) {
			outputs[row + i] += weights[(row + i) * numNodes + j] * inputs[j];
		}
		//outputs[row + i] *= (outputs[row + i] > 0.0f) * 1.1f - 0.1f;
	}
}

class Network {
public:
	uint64_t numNodes;
	uint64_t numWeights;
	uint64_t numThreads;
	uint64_t numNodesPerThread;
	uint64_t numRemainingNodes;
	uint64_t numPerseptions;
	uint64_t numActions;
	float* weights;
	float* biases;
	float* inputs;
	float* outputs;
	float* perceptions;
	float* actions;
	thread* threads;

	Network(uint64_t numNodes = 1000, uint64_t numThreads = 8, uint64_t numPerseptions = 2, uint64_t numActions = 1) {
		this->numNodes = numNodes;
		this->numWeights = numNodes * numNodes;
		this->numThreads = numThreads;
		this->numNodesPerThread = numNodes / numThreads;
		this->numRemainingNodes = numNodes - numNodesPerThread * numThreads;
		this->numPerseptions = numPerseptions;
		this->numActions = numActions;
		this->weights = new float[numWeights];
		this->biases = new float[numNodes];
		this->inputs = new float[numNodes];
		this->outputs = new float[numNodes];
		this->perceptions = new float[numPerseptions];
		this->actions = new float[numActions];
		this->threads = new thread[numThreads];
	}

	~Network() {
		delete[] weights;
		delete[] biases;
		delete[] inputs;
		delete[] outputs;
		delete[] perceptions;
		delete[] actions;
		delete[] threads;
	}

	Network(const Network& other) {
		this->numNodes = other.numNodes;
		this->numWeights = other.numWeights;
		this->numThreads = other.numThreads;
		this->numNodesPerThread = other.numNodesPerThread;
		this->numRemainingNodes = other.numRemainingNodes;
		this->numPerseptions = other.numPerseptions;
		this->numActions = other.numActions;
		this->weights = new float[numWeights];
		this->biases = new float[numNodes];
		this->inputs = new float[numNodes];
		this->outputs = new float[numNodes];
		this->perceptions = new float[numPerseptions];
		this->actions = new float[numActions];
		this->threads = new thread[numThreads];
		memcpy(this->weights, other.weights, numWeights * sizeof(float));
		memcpy(this->biases, other.biases, numNodes * sizeof(float));
		memcpy(this->inputs, other.inputs, numNodes * sizeof(float));
		memcpy(this->outputs, other.outputs, numNodes * sizeof(float));
		memcpy(this->perceptions, other.perceptions, numPerseptions * sizeof(float));
		memcpy(this->actions, other.actions, numActions * sizeof(float));
	}

	Network& operator=(const Network& other) {
		if (this != &other) {
			this->numNodes = other.numNodes;
			this->numWeights = other.numWeights;
			this->numThreads = other.numThreads;
			this->numNodesPerThread = other.numNodesPerThread;
			this->numRemainingNodes = other.numRemainingNodes;
			this->numPerseptions = other.numPerseptions;
			this->numActions = other.numActions;
			delete[] this->weights;
			delete[] this->biases;
			delete[] this->inputs;
			delete[] this->outputs;
			delete[] this->threads;
			this->weights = new float[numWeights];
			this->biases = new float[numNodes];
			this->inputs = new float[numNodes];
			this->outputs = new float[numNodes];
			this->perceptions = new float[numPerseptions];
			this->actions = new float[numActions];
			this->threads = new thread[numThreads];
			memcpy(this->weights, other.weights, numWeights * sizeof(float));
			memcpy(this->biases, other.biases, numNodes * sizeof(float));
			memcpy(this->inputs, other.inputs, numNodes * sizeof(float));
			memcpy(this->outputs, other.outputs, numNodes * sizeof(float));
			memcpy(this->perceptions, other.perceptions, numPerseptions * sizeof(float));
			memcpy(this->actions, other.actions, numActions * sizeof(float));
		}
		return *this;
	}

	Network(Network&& other) noexcept {
		this->numNodes = other.numNodes;
		this->numWeights = other.numWeights;
		this->numThreads = other.numThreads;
		this->numNodesPerThread = other.numNodesPerThread;
		this->numRemainingNodes = other.numRemainingNodes;
		this->numPerseptions = other.numPerseptions;
		this->numActions = other.numActions;
		this->weights = other.weights;
		this->biases = other.biases;
		this->inputs = other.inputs;
		this->outputs = other.outputs;
		this->perceptions = other.perceptions;
		this->actions = other.actions;
		this->threads = other.threads;
		other.weights = nullptr;
		other.biases = nullptr;
		other.inputs = nullptr;
		other.outputs = nullptr;
		other.perceptions = nullptr;
		other.actions = nullptr;
		other.threads = nullptr;
	}

	Network& operator=(Network&& other) noexcept {
		if (this != &other) {
			this->numNodes = other.numNodes;
			this->numWeights = other.numWeights;
			this->numThreads = other.numThreads;
			this->numNodesPerThread = other.numNodesPerThread;
			this->numRemainingNodes = other.numRemainingNodes;
			this->numPerseptions = other.numPerseptions;
			this->numActions = other.numActions;
			delete[] this->weights;
			delete[] this->biases;
			delete[] this->inputs;
			delete[] this->outputs;
			delete[] this->perceptions;
			delete[] this->actions;
			delete[] this->threads;
			this->weights = other.weights;
			this->biases = other.biases;
			this->inputs = other.inputs;
			this->outputs = other.outputs;
			this->perceptions = other.perceptions;
			this->actions = other.actions;
			this->threads = other.threads;
			other.weights = nullptr;
			other.biases = nullptr;
			other.inputs = nullptr;
			other.outputs = nullptr;
			other.perceptions = nullptr;
			other.actions = nullptr;
			other.threads = nullptr;
		}
		return *this;
	}

	void Initialize() {
		auto start = std::chrono::high_resolution_clock::now();
		uint64_t row = 0;
		for (uint64_t i = 0; i < numRemainingNodes; i++) {
			threads[i] = thread(initWeightsAndBiases, weights, biases, numNodes, row, numNodesPerThread + 1);
			row += numNodesPerThread + 1;
		}
		for (uint64_t i = numRemainingNodes; i < numThreads; i++) {
			threads[i] = thread(initWeightsAndBiases, weights, biases, numNodes, row, numNodesPerThread);
			row += numNodesPerThread;
		}
		for (int i = 0; i < numThreads; i++) {
			threads[i].join();
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		cout << "Elapsed time: " << elapsed.count() / 10 << " s" << endl;
	}

	void FeedForward() {
		auto start = std::chrono::high_resolution_clock::now();
		for (int itr = 0; itr < 10; itr++) {	// must be even for pointers to be back, or make param for itr and use if or smth
			uint64_t row = 0;
			for (uint64_t i = 0; i < numRemainingNodes; i++) {
				threads[i] = thread(multiplyWeightsAndBiases, weights, biases, inputs, outputs, perceptions, actions, numNodes, row, numNodesPerThread + 1);
				row += numNodesPerThread + 1;
			}
			for (uint64_t i = numRemainingNodes; i < numThreads; i++) {
				threads[i] = thread(multiplyWeightsAndBiases, weights, biases, inputs, outputs, perceptions, actions, numNodes, row, numNodesPerThread);
				row += numNodesPerThread;
			}
			for (int i = 0; i < numThreads; i++) {
				threads[i].join();
			}
			swap(inputs, outputs);
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		cout << "Elapsed time: " << elapsed.count() / 10 << " s" << endl;
	}

	void PrintParams() {
		for (uint64_t i = 0; i < numNodes; i++) {
			for (uint64_t j = 0; j < numNodes; j++) {
				cout << weights[i * numNodes + j] << " ";
			}
			cout << endl;
		}
		cout << endl;

		for (uint64_t i = 0; i < numNodes; i++) {
			cout << biases[i] << " ";
		}
		cout << endl << endl;
	}

	void PrintOutputs() {
		for (int i = 0; i < numNodes; i++) {
			cout << outputs[i] << " ";
		}
		cout << endl;
	}

	void LinearInput() {	// for testing until initialInput is added
		for (int i = 0; i < numNodes; i++) {
			inputs[i] = i;
		}
	}
};

int main() {
	Network net(100, 8);
	net.Initialize();
	net.PrintParams();
	net.LinearInput();
	net.FeedForward();
	net.PrintOutputs();

	return 0;
}