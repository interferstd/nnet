#include "neuron.h"
#include <cmath>


double Neuron::learning_rate = 0.15;
double Neuron::alpha = 0.5; // momentum
Activation Neuron::activation;

 Neuron::Neuron(unsigned numOutputs, unsigned index)
{
	outputWeights.reserve(numOutputs);
	for (unsigned c = 0; c < numOutputs; ++c)
	{
		outputWeights.push_back(Connection());
		outputWeights.back().weight = randomWeight();
	}

	index_ = index;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
	for (auto& neuron: prevLayer)
	{
		double oldDeltaWeight = neuron.outputWeights[index_].deltaWeight;

		double newDeltaWeight = learning_rate * neuron.getOutput() * gradient + alpha * oldDeltaWeight;
		neuron.outputWeights[index_].deltaWeight = newDeltaWeight;
		neuron.outputWeights[index_].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}

	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	gradient = dow * Neuron::activation.getDerivative()(output);
}

void Neuron::calcOutputGradients(double target)
{
	double delta = target - output;
	gradient = delta * Neuron::activation.getDerivative()(output);
}

void Neuron::predict(const Layer &prevLayer)
{
	double sum = 0.0;

	for (auto& neuron: prevLayer)
	{
		sum += neuron.getOutput() *
			neuron.outputWeights[index_].weight;
	}

	output = Neuron::activation.getFunction()(sum);
}

void Neuron::setDump(std::fstream& dumpFile)
{
	dumpFile << index_ << ' ' << outputWeights.size() << ' ';
	for (Connection& connect: outputWeights) dumpFile << connect.weight << ' ' << connect.deltaWeight << ' ';
	dumpFile << '\n';
}

void Neuron::getDump(std::fstream& loadFile)
{
	unsigned lenght;
	loadFile >> index_ >> lenght;
	outputWeights.resize(lenght);
	for (Connection& connect: outputWeights) loadFile >> connect.weight >> connect.deltaWeight;
}


double SigmoidFunction(double x)
{
	//output range [-1.0..1.0]
	return 1 / (1 + exp(-x));
}

double SigmoidDerivative(double x)
{
	double tmp = ::SigmoidFunction(x);
	return tmp * (1.0 - tmp);
}

void Activation::setSigmoid()
{
	Activation::set(::SigmoidFunction, ::SigmoidDerivative);
}

double TanhFunction(double x)
{
	//output range [-1.0..1.0]
	return tanh(x);
}

double TanhDerivative(double x)
{
	return 1.0 - x * x;
}

void Activation::setTanh()
{
	Activation::set(::TanhFunction, ::TanhDerivative);
}
