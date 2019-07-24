#pragma once
#include <vector>
#include <cstdlib>
#include <functional>

class Neuron;

typedef std::vector<Neuron> Layer;

typedef std::function<double(double)> Function;

struct Connection
{
	double weight;
	double deltaWeight;
};

class Activation{
public:
	Activation()
	{
		Activation::setTanh();
	}
	void set(Function function, Function derivative) { function_ = function; derivative_ = derivative; }
	inline Function getFunction() const { return function_; }
	inline Function getDerivative() const { return derivative_; }
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
private:
	static double learning_rate;
	static double alpha;
	static double randomWeight() { return std::rand() / double(RAND_MAX); }
	static Activation activation;
	double sumDOW(const Layer &nextLayer) const;
	double output;
	std::vector<Connection> outputWeights;
	unsigned index_;
	double gradient;
};
