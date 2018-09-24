import numpy as np
import pylab as plt
import networkx as nx

PLOT = True

paths_list = [(0,1),(1,5),(5,6),(5,4),(1,2),(2,3),(2,7)]
nodes_list = [0,1,2,3,4,5,6,7]
goal = 7

G=nx.Graph()
G.add_nodes_from(nodes_list)
G.add_edges_from(paths_list)
pos = nx.spring_layout(G)

def draw_Graph(CurrentState,Action,Q):
    node_color_List = ['r', 'r', 'r', 'r', 'r', 'r', 'r','r']
    node_color_List[CurrentState] = 'b'
    node_color_List[Action] = 'y'
    width_List = [Q[0,1], Q[1,5], Q[5,6], Q[5,4], Q[1,2], Q[2,3], Q[2,7]]
    if all(w == 0 for w in width_List):
        width_List = [1 for w in width_List]
    else:
        width_List = [w/max(width_List) for w in width_List]

    nx.draw_networkx_nodes(G,pos, node_color = node_color_List)
    nx.draw_networkx_edges(G,pos, width= width_List)
    nx.draw_networkx_labels(G,pos)
    plt.draw()


Matrix_Size = 8

R = np.matrix(np.ones(shape=(Matrix_Size,Matrix_Size)))
R *= -1


for path in paths_list:
    if path[1] == goal:
        R[path] = 100
    else:
        R[path] = 0
    if path[0] == goal:
        R[path[::-1]] = 100
    else:
        R[path[::-1]] = 0

R[goal,goal] = 100

print(R)


Q = np.matrix(np.zeros(([Matrix_Size,Matrix_Size])))

# learning parameter
gamma = 0.8

initial_state = 1

def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

available_act = available_actions(initial_state)

def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act,1))
    return next_action

action = sample_next_action(available_act)

def update(current_state, action, gamma):

  max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

  if max_index.shape[0] > 1:
      max_index = int(np.random.choice(max_index, size = 1))
  else:
      max_index = int(max_index)
  max_value = Q[action, max_index]

  Q[current_state, action] = R[current_state, action] + gamma * max_value
  print('max_value', R[current_state, action] + gamma * max_value)

  if (np.max(Q) > 0):
    return(np.sum(Q/np.max(Q)*100))
  else:
    return (0)

update(initial_state, action, gamma)

# Training
scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    score = update(current_state, action, gamma)
    if PLOT == True:
        draw_Graph(current_state, action, Q)
        print("Current State {}".format(current_state))
        print("Action {}".format(action))
        print(Q)
        plt.waitforbuttonpress()
    scores.append(score)
    print('Score:', str(score))

print("Trained Q matrix:")
print(Q / np.max(Q) * 100)


# Testing
current_state = 0
steps = [current_state]

while current_state != 7:

    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)

    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)

# plt.plot(scores)
# plt.show()