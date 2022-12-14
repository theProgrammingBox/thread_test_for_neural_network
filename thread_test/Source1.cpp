#include <thread>
#include <iostream>
#include "Randoms.h"

using std::thread;
using std::cout;
using std::endl;
using std::swap;

void initParams(float* initialInputs, float* weights, float* biases, uint64_t numNodes, uint64_t numInputs, uint64_t currentNode, uint16_t nodes) {
	Random r;
	for (uint64_t i = currentNode; i < currentNode + nodes; i++) {
		initialInputs[i * (i < numNodes)] = r.normalRand() * 0.01f;
		for (uint64_t j = 0; j < numInputs; j++) {
			weights[i * numInputs + j] =  ((i == j) && (j < numNodes)) + 0.01f * r.normalRand();
		}
		biases[i] = r.normalRand() * 0.01f;
	}
}

void multiplyWeightsAndBiases(float* weights, float* biases, float* inputs, float* outputs, uint64_t numInputs, uint64_t currentNode, uint16_t nodes) {
	for (uint64_t i = currentNode; i < currentNode + nodes; i++) {
		outputs[i] = biases[i];
		for (uint64_t j = 0; j < numInputs; j++) {
			outputs[i] += weights[i * numInputs + j] * inputs[j];
		}
		outputs[i] = (outputs[i] > 1) - (outputs[i] < -1) + !(outputs[i] > 1 || outputs[i] < -1) * outputs[i];
	}
}

class Network {						// only input and output layer, think of it like a brain's state of mind changing over time
public:
	uint64_t numNodes;				// number of purely reccursive nodes in the network, the "brain"
	uint64_t numPerseptions;		// number of inputs to the network, the "senses"
	uint64_t numActions;			// number of outputs from the network, the "actions"
	uint64_t numInputs;				// number of inputs to the network, contains the perceptions and the outputs or "brain" of the previous iteration
	uint64_t numOutputs;			// number of outputs from the network, contains the actions and the outputs or changed "brain" from the current iteration
	uint64_t numWeights;			// number of weights in the network
	float* inputs;					// inputs to the network, contains the perceptions and the outputs or "brain" of the previous iteration
	float* outputs;					// outputs from the network, contains the actions and the outputs or changed "brain" from the current iteration
	float* initialInputs;			// initial state of mind during "birth"
	float* weights;					// weights of the network
	float* biases;					// biases of the network

	uint64_t numThreads;			// for parallelization
	uint64_t numNodesPerThread;		// minimum number of nodes handled per thread
	uint64_t numRemainingNodes;		// number of threads handling one more node
	thread* threads;				// array of threads

	Network(uint64_t numNodes = 1000, uint64_t numPerseptions = 2, uint64_t numActions = 1, uint64_t numThreads = 4) {
		this->numNodes = numNodes;
		this->numPerseptions = numPerseptions;
		this->numActions = numActions;
		this->numInputs = numPerseptions + numNodes;
		this->numOutputs = numActions + numNodes;
		this->numWeights = numInputs * numOutputs;
		this->inputs = new float[numInputs];
		this->outputs = new float[numOutputs];
		this->initialInputs = new float[numNodes];
		this->weights = new float[numWeights];
		this->biases = new float[numOutputs];
		this->numThreads = numThreads;
		this->numNodesPerThread = numOutputs / numThreads;
		this->numRemainingNodes = numOutputs - numNodesPerThread * numThreads;
		this->threads = new thread[numThreads];
	}

	~Network() {
		delete[] inputs;
		delete[] outputs;
		delete[] initialInputs;
		delete[] weights;
		delete[] biases;
		delete[] threads;
	}

	Network(const Network& other) {
		numNodes = other.numNodes;
		numPerseptions = other.numPerseptions;
		numActions = other.numActions;
		numInputs = other.numInputs;
		numOutputs = other.numOutputs;
		numWeights = other.numWeights;
		inputs = new float[numInputs];
		outputs = new float[numOutputs];
		initialInputs = new float[numNodes];
		weights = new float[numWeights];
		biases = new float[numOutputs];
		numThreads = other.numThreads;
		numNodesPerThread = other.numNodesPerThread;
		numRemainingNodes = other.numRemainingNodes;
		threads = new thread[numThreads];
		memcpy(inputs, other.inputs, numInputs * sizeof(float));
		memcpy(outputs, other.outputs, numOutputs * sizeof(float));
		memcpy(initialInputs, other.initialInputs, numNodes * sizeof(float));
		memcpy(weights, other.weights, numWeights * sizeof(float));
		memcpy(biases, other.biases, numOutputs * sizeof(float));
	}

	Network& operator=(const Network& other) {
		if (this != &other) {
			numNodes = other.numNodes;
			numPerseptions = other.numPerseptions;
			numActions = other.numActions;
			numInputs = other.numInputs;
			numOutputs = other.numOutputs;
			numWeights = other.numWeights;
			delete[] inputs;
			delete[] outputs;
			delete[] initialInputs;
			delete[] weights;
			delete[] biases;
			inputs = new float[numInputs];
			outputs = new float[numOutputs];
			initialInputs = new float[numNodes];
			weights = new float[numWeights];
			biases = new float[numOutputs];
			numThreads = other.numThreads;
			numNodesPerThread = other.numNodesPerThread;
			numRemainingNodes = other.numRemainingNodes;
			delete[] threads;
			threads = new thread[numThreads];
			memcpy(inputs, other.inputs, numInputs * sizeof(float));
			memcpy(outputs, other.outputs, numOutputs * sizeof(float));
			memcpy(initialInputs, other.initialInputs, numNodes * sizeof(float));
			memcpy(weights, other.weights, numWeights * sizeof(float));
			memcpy(biases, other.biases, numOutputs * sizeof(float));
		}
		return *this;
	}

	Network(Network&& other) noexcept {
		numNodes = other.numNodes;
		numPerseptions = other.numPerseptions;
		numActions = other.numActions;
		numInputs = other.numInputs;
		numOutputs = other.numOutputs;
		numWeights = other.numWeights;
		inputs = other.inputs;
		outputs = other.outputs;
		initialInputs = other.initialInputs;
		weights = other.weights;
		biases = other.biases;
		numThreads = other.numThreads;
		numNodesPerThread = other.numNodesPerThread;
		numRemainingNodes = other.numRemainingNodes;
		threads = other.threads;
		other.inputs = nullptr;
		other.outputs = nullptr;
		other.initialInputs = nullptr;
		other.weights = nullptr;
		other.biases = nullptr;
		other.threads = nullptr;
	}

	Network& operator=(Network&& other) noexcept {
		if (this != &other) {
			numNodes = other.numNodes;
			numPerseptions = other.numPerseptions;
			numActions = other.numActions;
			numInputs = other.numInputs;
			numOutputs = other.numOutputs;
			numWeights = other.numWeights;
			delete[] inputs;
			delete[] outputs;
			delete[] initialInputs;
			delete[] weights;
			delete[] biases;
			delete[] threads;
			inputs = other.inputs;
			outputs = other.outputs;
			initialInputs = other.initialInputs;
			weights = other.weights;
			biases = other.biases;
			numThreads = other.numThreads;
			numNodesPerThread = other.numNodesPerThread;
			numRemainingNodes = other.numRemainingNodes;
			threads = other.threads;
			other.inputs = nullptr;
			other.outputs = nullptr;
			other.initialInputs = nullptr;
			other.weights = nullptr;
			other.biases = nullptr;
			other.threads = nullptr;
		}
		return *this;
	}

	void Initialize() {
		auto start = std::chrono::high_resolution_clock::now();
		uint64_t currentOutput = 0;
		for (uint64_t i = 0; i < numRemainingNodes; i++) {
			threads[i] = thread(initParams, initialInputs, weights, biases, numNodes, numInputs, currentOutput, numNodesPerThread + 1);
			currentOutput += numNodesPerThread + 1;
		}
		for (uint64_t i = numRemainingNodes; i < numThreads; i++) {
			threads[i] = thread(initParams, initialInputs, weights, biases, numNodes, numInputs, currentOutput, numNodesPerThread);
			currentOutput += numNodesPerThread;
		}
		for (int i = 0; i < numThreads; i++) {
			threads[i].join();
		}
		memcpy(inputs, initialInputs, numNodes * sizeof(float));
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		cout << "Elapsed time: " << elapsed.count() / 10 << " s" << endl;
	}

	void FeedForward(float* input, float* output) {
		auto start = std::chrono::high_resolution_clock::now();
		memcpy(inputs + numNodes, input, numPerseptions * sizeof(float));
		for (int itr = 0; itr < 10; itr++) {
			PrintOutputs();
			uint64_t currentOutput = 0;
			for (uint64_t i = 0; i < numRemainingNodes; i++) {
				threads[i] = thread(multiplyWeightsAndBiases, weights, biases, inputs, outputs, numInputs, currentOutput, numNodesPerThread + 1);
				currentOutput += numNodesPerThread + 1;
			}
			for (uint64_t i = numRemainingNodes; i < numThreads; i++) {
				threads[i] = thread(multiplyWeightsAndBiases, weights, biases, inputs, outputs, numInputs, currentOutput, numNodesPerThread);
				currentOutput += numNodesPerThread;
			}
			for (int i = 0; i < numThreads; i++) {
				threads[i].join();
			}
			memcpy(inputs, outputs, numNodes * sizeof(float));
		}
		memcpy(output, outputs + numNodes, numActions * sizeof(float));
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		cout << "Elapsed time: " << elapsed.count() / 10 << " s" << endl;
	}

	void PrintParams() {
		cout << "Weights:\n";
		for (uint64_t i = 0; i < numOutputs; i++) {
			for (uint64_t j = 0; j < numInputs; j++) {
				cout << weights[i * numInputs + j] << " ";
			}
			cout << endl;
		}
		cout << endl;

		cout << "Biases:\n";
		for (uint64_t i = 0; i < numOutputs; i++) {
			cout << biases[i] << " ";
		}
		cout << endl << endl;

		cout << "Initial Inputs:\n";
		for (uint64_t i = 0; i < numNodes; i++) {
			cout << initialInputs[i] << " ";
		}
		cout << endl << endl;
	}

	void PrintOutputs() {
		/*cout << "Input:\n";
		for (uint64_t i = 0; i < numInputs; i++) {
			cout << inputs[i] << " ";
		}
		cout << endl << endl;*/

		cout << "Outputs:\n";
		for (uint64_t i = 0; i < numOutputs; i++) {
			cout << outputs[i] << " ";
		}
		cout << endl << endl;
	}
};

int main() {
	float* input = new float[2]{ 0, 1 };
	float* output = new float[1]{ 0 };
	Network net(10, 2, 1, 10);
	net.Initialize();
	net.PrintParams();
	net.FeedForward(input, output);
	net.PrintOutputs();

	return 0;
}