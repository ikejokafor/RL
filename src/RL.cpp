#include "RL.hpp"
using namespace std;

QLearner::QLearner(int numStates, bool *validActionsPerState, float discountFactor, float learningRate, float epsilon, int *startState) {
	m_numStates = numStates;
	m_validActionsPerState = (bool*)malloc(numStates * numStates * sizeof(bool));
	memcpy(m_validActionsPerState, validActionsPerState, (numStates *numStates * sizeof(bool)));
	m_discountFactor = discountFactor;
	m_learningRate = learningRate;
	m_epsilon = epsilon;
	m_QMatrix = (float*)malloc(numStates * numStates * sizeof(float));
	memset(m_QMatrix, 0, (numStates * numStates * sizeof(float)));
	
	if(startState == NULL) {
		m_currentState = 0;
	} else {
		m_currentState = startState[0];
	}
	

	m_gen = new default_random_engine();
	m_dis = new uniform_real_distribution<float>(0.0f, 1.0f);
}


QLearner::~QLearner() {
    free(m_QMatrix);
	free(m_validActionsPerState);
}


int QLearner::GetNextAction() {
	if((m_dis[0](m_gen[0])) <= m_epsilon) {
		srand(time(NULL));
		while(true) {
			m_currentAction = rand() % m_numStates;
			if (index2D(m_numStates, m_numStates, m_validActionsPerState, m_currentState, m_currentAction)) {
				m_epsilon -= 0.001f;
				return m_currentAction;
			}
		}
	} else {
		m_currentAction = GetBestAction(m_currentState);
		m_epsilon -= 0.001f;
		return m_currentAction;
	}
}


int QLearner::GetBestAction(int state) {
	vector<int> actionList;
	for (int i = 0; i < m_numStates; i++) {
		if (index2D(m_numStates, m_numStates, m_validActionsPerState, state, i)) {
			if (actionList.size() == 0) {
				actionList.push_back(i);
			} else if(index2D(m_numStates, m_numStates, m_QMatrix, state, i) == m_QMatrix[actionList[0]]) {
				actionList.push_back(i);
			} else if(index2D(m_numStates, m_numStates, m_QMatrix, state, i) > m_QMatrix[actionList[0]]) {
				actionList.clear();
				actionList.push_back(i);
			}
		}
	}
	
	if (actionList.size() == 1) {
		return actionList[0];
	} else {
		srand(time(NULL));
		return actionList[rand() % actionList.size()];
	}
}


void QLearner::UpdateQTable(float reward) {
	// States and actions have a one to one mapping; ie state 0 can only do action 0, state 1 can only do action 1, so state id is same as action id
	index2D(m_numStates, m_numStates, m_QMatrix, m_currentState, m_currentAction)
		+= (reward + m_discountFactor * index2D(m_numState, m_numStates, m_QMatrix, m_currentAction, GetBestAction(m_currentAction)));
	m_currentState = m_currentAction;
	printQMatrix();
}


void QLearner::printQMatrix() {
	for(int i = 0; i < m_numStates; i++) {
		for (int j = 0; j < m_numStates; j++) {
			cout << index2D(m_numStates, m_numStates, m_QMatrix, i, j) << " ";
		}
		cout << endl;
	}
}