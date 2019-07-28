#pragma once
#include <vector>
#include <cstdlib>
#include <functional>
#include <fstream>

class Neuron;

typedef std::vector<Neuron> Layer;

typedef std::function<double(double)> Function;

typedef struct
{
	double weight;
	double deltaWeight;
} Connection;

class Activation{
public:
	Activation()
	{
		Activation::setTanh();
	}
	void set(Function function, Function derivative) { function_ = function; derivative_ = derivative; }
	inline const Function& getFunction() { return function_; }
	inline const Function& getDerivative() { return derivative_; }
	void setSigmoid();
	void setTanh();
private:
	Function function_;
	Function derivative_;
};

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutput(double val) { output = val; }
	double getOutput() const { return output; }
	void predict(const Layer &prevLayer);
	void calcOutputGradients(double targetVals);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	void setDump(std::fstream& dumpFile);
	void getDump(std::fstream& loadFile);
	static void setLearnRate(double learningRate = 0.15) { learning_rate = learningRate; }
	static void setAlpha(double newAlpha = 0.5) { alpha = newAlpha; }
	static Activation activation;
private:
	static double learning_rate;
	static double alpha;
	static double randomWeight() { return std::rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double output;
	std::vector<Connection> outputWeights;
	unsigned index_;
	double gradient;
};
