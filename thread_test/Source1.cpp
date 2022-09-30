#include <thread>
#include <iostream>
#include <chrono>

using std::thread;
using std::cout;
using std::endl;

// thread function to initialize weights and biases
void initWeightsAndBiases(float* weights, float* biases, uint64_t numNodes, uint64_t row, uint16_t numRows) {
	for (uint64_t i = 0; i < numRows; i++) {
		for (uint64_t j = 0; j < numNodes; j++) {
			weights[(row + i) * numNodes + j] = (row + i) == j;
		}
		biases[row + i] = 0.0f;
	}
}

// thread function to multiply weights and biases
void multiplyWeightsAndBiases(float* inputs, float* weights, float* biases, float* outputs, uint64_t numNodes, uint64_t row, uint16_t numRows) {
	// outputs = weights * inputs + biases
	for (uint64_t i = 0; i < numRows; i++) {
		outputs[row + i] = biases[row + i];
		for (uint64_t j = 0; j < numNodes; j++) {
			outputs[row + i] += weights[(row + i) * numNodes + j] * inputs[j];
		}
		//outputs[row + i] *= (outputs[row + i] > 0.0f) * 1.1f - 0.1f;
	}
}

int main() {
	uint64_t numNodes = 10000;
	uint64_t numWeights = numNodes * numNodes;
	uint64_t numThreads = 8;
	uint64_t numNodesPerThread = numNodes / numThreads;
	uint64_t numRemainingNodes = numNodes - numNodesPerThread * numThreads;
	
	float* weights = new float[numWeights];
	float* biases = new float[numNodes];
	float* inputs = new float[numNodes];
	float* outputs = new float[numNodes];
	thread* threads = new thread[numThreads];


	/*cout << numNodesPerThread << endl;
	cout << numRemainingNodes << endl;*/

	auto start = std::chrono::high_resolution_clock::now();
	for (int itr = 0; itr < 10; itr++) {
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
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	cout << "Elapsed time: " << elapsed.count() / 10 << " s" << endl;

	/*for (uint64_t i = 0; i < numNodes; i++) {
		for (uint64_t j = 0; j < numNodes; j++) {
			cout << weights[i * numNodes + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	for (uint64_t i = 0; i < numNodes; i++) {
		cout << biases[i] << " ";
	}
	cout << endl << endl;*/

	for (int i = 0; i < numNodes; i++) {
		inputs[i] = i;
	}

	start = std::chrono::high_resolution_clock::now();
	for (int itr = 0; itr < 10; itr++) {
		uint64_t row = 0;
		for (uint64_t i = 0; i < numRemainingNodes; i++) {
			threads[i] = thread(multiplyWeightsAndBiases, inputs, weights, biases, outputs, numNodes, row, numNodesPerThread + 1);
			row += numNodesPerThread + 1;
		}
		for (uint64_t i = numRemainingNodes; i < numThreads; i++) {
			threads[i] = thread(multiplyWeightsAndBiases, inputs, weights, biases, outputs, numNodes, row, numNodesPerThread);
			row += numNodesPerThread;
		}

		for (int i = 0; i < numThreads; i++) {
			threads[i].join();
		}
	}
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	cout << "Elapsed time: " << elapsed.count() / 10 << " s" << endl;

	/*for (int i = 0; i < numNodes; i++) {
		cout << outputs[i] << " ";
	}
	cout << endl;*/

	delete[] weights;
	delete[] biases;
	delete[] inputs;
	delete[] outputs;
	delete[] threads;
	return 0;
}