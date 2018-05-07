#ifndef __RL_HPP__
#define __RL_HPP__


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>
#include <vector>
#include <ctime>
#include <iostream>

#include "util.hpp"

class QLearner {
    public:
		QLearner(int numStates, bool *numActionsPerState, float discountFactor, float learningRate, float epsilon, int *startState);
		~QLearner();
		int GetNextAction();
		int GetBestAction(int state);
		void UpdateQTable(float reward);
		void printQMatrix();
        
        float *m_QMatrix;
        float m_discountFactor;
		float m_learningRate;
		float m_epsilon;
		int m_currentState;
		int m_numStates;
		int m_currentAction;
		int m_prevState;
		bool *m_validActionsPerState;
		std::default_random_engine *m_gen;
		std::uniform_real_distribution<float> *m_dis;
	
    protected:
    
    private:


};

#endif