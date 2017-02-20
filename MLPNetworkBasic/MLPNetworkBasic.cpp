// MLPNetworkBasic.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <vector>
#include <functional>
#include <cmath>
#include <stdlib.h>
#include <chrono>
#include <thread>

std::vector<std::vector<double>> GetRandomCase()
{
	double X = ((double)std::rand() - (double)RAND_MAX / 2) / (double)RAND_MAX;
	double Y = ((double)std::rand() - (double)RAND_MAX / 2) / (double)RAND_MAX;
	X *= 20;
	Y *= 20;

	std::vector<std::vector<double>> RetVec;
	std::vector<double> Values = { X , Y };
	std::vector<double> Result;
	if ((abs(X) < 5) && (abs(Y) < 5))
	{Result.push_back(1);}
	else { Result.push_back(0); }
	RetVec.push_back(Values);
	RetVec.push_back(Result);

	return RetVec;
}


class TheNetwork
{
public:

	std::vector<double> GetLastResult()
	{
		return Results[Results.size()-1];
	}

	void setActivationalFunc(std::function<double(double)> _ativationalFunc)
	{ativationalFunc = _ativationalFunc;}

	void setSizes(int _sizeOfInput, int _sizeOfHiddenLevels, int _sizeOfOutput, int _NetworkLength)
	{
		sizeOfHiddenLevels = _sizeOfHiddenLevels;
		sizeOfInput = _sizeOfInput;
		sizeOfOutput = _sizeOfOutput;
		NetworkLength = _NetworkLength;
	}

	void SetWeights(std::vector<std::vector<std::vector<double>>> _Weights)
	{
		Weights = _Weights;
	}

	std::vector<double> *calculateOutput(std::vector<double> input)
	{
		//Check the number of values in input
		if (input.size() > sizeOfInput)		{return nullptr;}

		//Add bias to the input
		std::vector<double> biasedInput;
		biasedInput.push_back(1);
		for (auto i = input.begin(); i != input.end(); i++)
		{biasedInput.push_back(*i);	}
		
		//Delete results from last calc;
		Results.clear();

		//Fit in input
		Results.push_back(biasedInput);

	
		//Get the rest values;
			//Starts from one, becouse 0 is the input
		for (int tmpLevel = 1; tmpLevel <= NetworkLength; tmpLevel++)
		{
			//Create temporary storage
			std::vector<double> tmpStorage;
			//Just in case
			tmpStorage.clear();

			//Add a bias
			tmpStorage.push_back(1);

			//Check for the output neurons. Thay can me in a diffrent number, then the rest network
			int sizeOfCurrentLevel = sizeOfHiddenLevels;
			if(tmpLevel == NetworkLength)		{sizeOfCurrentLevel = sizeOfOutput;	}

			//starts from 1, becouse 0 is a bias			
			for (int tmpNumber = 1; tmpNumber <= sizeOfCurrentLevel; tmpNumber++)
			{

				/*We checking, if this is a first iteration, becouse there might be diffrent
				amount of inputs for the first level of hidden neurons(right after the input)
				or for the rest of the network */
				int neuronsInLevel=0;
				if (tmpLevel == 1)	{ neuronsInLevel = sizeOfInput;}
				else { neuronsInLevel = sizeOfHiddenLevels;}

				double sum = 0;
				//Summate the input
				for (int neuronInputNumber = 0; neuronInputNumber < neuronsInLevel; neuronInputNumber++)
				{
					//Weights count neurons from 0; Deal with it
					sum += Results[tmpLevel - 1][neuronInputNumber] * Weights[tmpLevel][tmpNumber-1][neuronInputNumber];
				}
				//Calc the result and put it in storage
				tmpStorage.push_back(ativationalFunc(sum));
			}

			//Move contents of storate in results;
			Results.push_back(std::move(tmpStorage));
		}
		
		//Send results of output level, except bias, out
		std::vector<double> *ReturnValue = new std::vector<double>;
		for (int i = 1; i < Results[NetworkLength].size(); i++)
		{
			ReturnValue->push_back(Results[NetworkLength][i]);
		}
		return ReturnValue;
	}

	double doABackpropagetion(std::vector<double> testInput, std::vector<double> expectedOutput, double lerningParam)
	{
		//Evaluate the network and drop the actual results. We need only the inner Results to be rebuild
		std::vector<double> *actualOutput;
		actualOutput = calculateOutput(testInput);
		delete actualOutput;

		//Create the errors matrix, with size of network
		std::vector<std::vector<double>> ErrorsMatrix;
		ErrorsMatrix.resize(NetworkLength);


		//Now we descending to backpropagete errors
		for (int tmpLevel = NetworkLength; tmpLevel > 0; tmpLevel--)
		{

			int sizeOfCurrentLevel = sizeOfHiddenLevels;
			if (tmpLevel == NetworkLength) { sizeOfCurrentLevel = sizeOfOutput; }

			for (int NeuronNumber = 1; NeuronNumber <= sizeOfCurrentLevel; NeuronNumber++)
			{

				//The error calc is different for output neurons and the rest of the network
				double tmpError = 0;
				if (tmpLevel == NetworkLength)
				{
					tmpError = Results[tmpLevel][NeuronNumber] * (1 - Results[tmpLevel][NeuronNumber])*(expectedOutput[NeuronNumber - 1] - Results[tmpLevel][NeuronNumber]);
				}
				else
				{
					for (int UpperLevelNeuronNum = 0; UpperLevelNeuronNum < ErrorsMatrix[tmpLevel].size(); UpperLevelNeuronNum++)
					{
						//The ErrorMatrix starts with 0, but we also have more lavels then should, so it does not affect
						tmpError = Weights[tmpLevel + 1][UpperLevelNeuronNum][NeuronNumber] * ErrorsMatrix[tmpLevel][UpperLevelNeuronNum];
					}
					tmpError *= Results[tmpLevel][NeuronNumber] * (1 - Results[tmpLevel][NeuronNumber]);

				}

				ErrorsMatrix[tmpLevel-1].push_back(tmpError);
			}
		}

		//and now we ajust all weights!

		for (int Level = 1; Level < Weights.size(); Level++)
		{
			for (int NeuronNum = 0; NeuronNum < Weights[Level].size(); NeuronNum++)
			{
				for (int ConnectionNum = 0; ConnectionNum < Weights[Level][NeuronNum].size(); ConnectionNum++)
				{
					Weights[Level][NeuronNum][ConnectionNum] += lerningParam*Results[Level - 1][ConnectionNum] * ErrorsMatrix[Level-1][NeuronNum];
				}
			}
		}
		
		double OverallQuadraticError = 0;
		for (int i = 0; i < ErrorsMatrix[NetworkLength-1].size(); i++)
		{
			OverallQuadraticError += ErrorsMatrix[NetworkLength-1][i] * ErrorsMatrix[NetworkLength-1][i];
		}
		return OverallQuadraticError;
	}

	void generateRandomWeights()
	{
		Weights.clear();

		Weights.resize(NetworkLength + 1);
		for (int Level = 1; Level < NetworkLength + 1; Level++)
		{

			int CurrentLevelSize = sizeOfHiddenLevels;
			if (Level == NetworkLength) { CurrentLevelSize = sizeOfOutput; }

			Weights[Level].resize(CurrentLevelSize);
			for (int tmpNeuron = 0; tmpNeuron < CurrentLevelSize; tmpNeuron++)
			{
				int CurrentNewronConnectionsAmount = sizeOfHiddenLevels+1;
				if (Level == 1) { CurrentNewronConnectionsAmount = sizeOfInput+1; }

				Weights[Level][tmpNeuron].resize(CurrentNewronConnectionsAmount);
				for (int j = 0; j < CurrentNewronConnectionsAmount; j++)
				{
					Weights[Level][tmpNeuron][j] = ((double)std::rand() - (double)RAND_MAX / 2) / (double)RAND_MAX;
				}
			}

		}
			

			

	}

private:

	std::function<double(double)> ativationalFunc;

	int sizeOfHiddenLevels;//The number of neurons in each hidden level
	int sizeOfInput;
	int sizeOfOutput;
	int NetworkLength;

	/*First is the level of network, second is the number in level and third is number of neuron in previous
	level, which sends the data. And also number zero is reserved for bias*/
	std::vector<std::vector<std::vector<double>>> Weights;

	/*Perceptrons defined by its level (0-input; NetworkLength-Output;)
	and its position in the	level (from 1 up to sizeOfInput/sizeOfHiddenLevels/sizeOfOutput)*/
	double getPerceptronOutput(int level, int number)
	{
		return 0;
	}

	double getConnectionWeight(int fromLevel, int fromNumber, int toLevel, int toNumber)
	{
		return 0;
	}
	
	//Results of the last calculation
	//Rules are the same [Level, positon]
	std::vector<std::vector<double>> Results;



};

int main()
{
	TheNetwork Network;
	Network.setSizes(2, 10, 1, 3);

	//For now THIS IS A HARDOCODED FUNCTION. It's derevitive is hardcoded into backprop, becouse
	//otherwise it'll have to store not only the results of neurons, but also their weighted sum of inputs,
	//which wil ether take the evaluetion of all the network, or an additional storage.
	Network.setActivationalFunc([](double x) {return (1 / (1 + exp(-x))); });

	//Weights;
	Network.generateRandomWeights();
	/*std::vector<std::vector<std::vector<double>>> Weights = 
	{ 
		{ {1} },
		{ {1,1,1}, {1,1,1}, { 1,1,1 }, { 1,1,1 },{ 1,1,1 },{ 1,1,1 },{ 1,1,1 },{ 1,1,1 },{ 1,1,1 },{ 1,1,1 }, },
		{ {1,2,2,2,2,2,2,2,2,2,2},{ 1,2,2,2,2,2,2,2,2,2,2 } }
	};*/

	//Network.SetWeights(Weights);

	std::vector<double> *Results;
	Results = Network.calculateOutput(std::vector<double> {1, 1});

	int Counter = 0;

	std::vector<std::vector<std::vector<double>>> Cases = {
		{ { 0,0 },{ 1 } },
		{ { 2,3 },{ 1 } },
		{ { 5,5 },{ 1 } },
		{ { 6,6 },{ 0 } },
		{ { 8,11},{ 0 } },
		{ { 4,1 },{ 1 } },
		{ { 2,12 },{ 0 } },
		{ { 40,1 },{ 0 } },
		{ { 0,1 },{ 1 } },
		{ { 13,42 },{ 0 } },

	};

	while (Counter < 10)
	{
		double Error;

		int RightAns = 0;
		for(int Step = 0; Step < 100; Step++)
		{
			//std::vector<std::vector<double>> Case = Cases[Counter % 10];
			std::vector<std::vector<double>> Case = GetRandomCase();

			Error = Network.doABackpropagetion(Case[0], Case[1], 0.4);

			double Result = Network.GetLastResult()[1];
			if (((Result < 0.5) && (Case[1][0] < 0.5)) || ((Result >= 0.5) && (Case[1][0] >= 0.5)))
			{
				RightAns++;
			}

		}
		printf("Correct [%3i] Counter = %i\n", RightAns, Counter);

		//std::vector<std::vector<double>> Case = Cases[Counter % 10];
		//Error = Network.doABackpropagetion(Case[0], Case[1], 0.4);

		//printf("So on %ith iteration with case [[% 6.4f, % 6.4f]-> % 6.4f] Network thinks its [ %6.4f], netwok did % 6.4f error\n", Counter, Case[0][0], Case[0][1], Case[1][0],Network.GetLastResult()[1], Error);
		Counter++;
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		
	}

	printf("Final check\n");

	int RightAns = 0;
	for (int Step = 0; Step < 100; Step++)
	{
		//std::vector<std::vector<double>> Case = Cases[Counter % 10];
		std::vector<std::vector<double>> Case = GetRandomCase();

		Network.calculateOutput(Case[0]);

		double Result = Network.GetLastResult()[1];
		if (((Result < 0.5) && (Case[1][0] < 0.5)) || ((Result >= 0.5) && (Case[1][0] >= 0.5)))
		{
			RightAns++;
		}

	}
	printf("Correct [%3i] Counter = %i\n", RightAns, Counter);

    return 0;
}

