from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from itertools import combinations
from typing import List

BUDGET = 500_000
SMARTPHONE_COST = 15_000
FRIEND_COST = 100
REWARD = 700

def get_attrs(G: nx.Graph, first=False, attrs={}):
    """Function for initializing and updating nodes attributes.
    Args:
        G: nx graph
        first: initialize if True, else update
        attrs: dict of nodes attributes
    """
    for node in G.nodes():
        if first == True:  # инициализируем нужные аттрибуты для вершин
            friends = len([n for n in G.neighbors(node)])
            price = SMARTPHONE_COST + FRIEND_COST * friends

            attrs[node] = {
                "price": price,
                "friends": friends,
                "friends_bought": 0,
                "marked": 0
            }
        else:  # проходимся по всем соседям, пересчитываем сколько из них купили, сохраняем в аттрибут
            fb = 0
            for nei in [n for n in G.neighbors(node)]:
                if G.nodes[nei]["marked"] != 0:
                    fb += 1
            attrs[node]["friends_bought"] = fb

    return attrs

def test_activation_list(G: nx.Graph, activation_list: List):
    G = G.copy()
    attrs = get_attrs(G, first=True)

    nx.set_node_attributes(G, attrs)

    player_budg = [BUDGET]
    player_users = [0]

    answer_nodes = []

    # проводится 180 шагов симуляции
    for _ in tqdm(range(180)):
        step_price = 0
        step_reward = 0
        step_users = 0
        update_today = []

        for n in activation_list:  # проход по всем вершинам в списке на активации
            if G.nodes[n]["marked"] == 0:  # берём первую, ещё неактивированную
                step_price += G.nodes[n]["price"]
                step_users += 1
                G.nodes[n]["marked"] = 1
                answer_nodes.append(n)
                break

        UPDATE = True
        while UPDATE:
            UPDATE = False  # (???)
            for n in G.nodes():  # проходим по всем вершинам в графе
                fb = 0
                for nei in [n for n in G.neighbors(n)]:  # подсчитываем зараженных соседей
                    if G.nodes[nei]["marked"] == 1:
                        fb += 1
                G.nodes[n]["friends_bought"] = fb

                if G.nodes[n]["friends_bought"] * 5 >= G.nodes[n]["friends"] and G.nodes[n]["marked"] == 0:
                    G.nodes[n]["marked"] = 2
                    update_today.append(n)
                    step_users += 1
                    step_reward += REWARD


        for n in update_today:
            G.nodes[n]["marked"] = 1

        player_budg.append(player_budg[-1] - step_price + step_reward)
        player_users.append(player_users[-1] + step_users)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.set_title("money")
    ax1.plot(player_budg)

    ax2.set_title("users")
    ax2.plot(player_users)

    plt.show()
    print('Max budget:', max(player_budg))
    print('Max budget step:', np.argmax(player_budg))
    return player_budg

def get_activation_list(G: nx.Graph):
    G = G.copy()
    
    centralities = nx.betweenness_centrality(G, k=10, seed=0)
    activation_list = list(G.nodes)
    activation_list.sort(key=lambda x: centralities[x], reverse=True)

    attrs = get_attrs(G, first=True)
    nx.set_node_attributes(G, attrs)

    player_budg = [BUDGET]
    player_users = [0]
    activated = []
    best_set = []
    max_revenue = 0

    # проводится 180 шагов симуляции
    for step in tqdm(range(180)):
        step_price = 0
        step_reward = 0
        step_users = 0
        update_today = []

        for n in activation_list:  # проход по всем вершинам в списке на активации
            if G.nodes[n]["marked"] == 0:  # берём первую, ещё неактивированную
                step_price += G.nodes[n]["price"]
                step_users += 1
                G.nodes[n]["marked"] = 1
                activated.append(n)
                break

        for n in G.nodes():  # проходим по всем вершинам в графе
            fb = 0
            for nei in [n for n in G.neighbors(n)]:  # подсчитываем зараженных соседей
                if G.nodes[nei]["marked"] == 1:
                    fb += 1
            G.nodes[n]["friends_bought"] = fb

            if G.nodes[n]["friends_bought"] * 5 >= G.nodes[n]["friends"] and G.nodes[n]["marked"] == 0:
                G.nodes[n]["marked"] = 2
                update_today.append(n)
                step_users += 1
                step_reward += REWARD

        for n in update_today:
            G.nodes[n]["marked"] = 1

        player_budg.append(player_budg[-1] - step_price + step_reward)
        player_users.append(player_users[-1] + step_users)

        if (step + 1) % 10 == 0:
            activation_list_new = update_activation_list(G)
            if activation_list_new is not None:
                activation_list = activation_list_new

        free_spread_revenue = free_spread(G, step + 1, player_budg[-1])
        if free_spread_revenue > max_revenue:
            best_set = activated.copy()
            max_revenue = free_spread_revenue

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # ax1.set_title("money")
    # ax1.plot(player_budg)

    # ax2.set_title("users")
    # ax2.plot(player_users)

    # plt.show()
    # print('Max budget:', max(player_budg))
    # print('Max budget step:', np.argmax(player_budg))
    return best_set

def update_activation_list(G: nx.Graph):
    # надо дропать все помеченные и взвешивать
    G = G.copy()
    to_remove = []
    attrs = {}
    for node in G.nodes:
        if G.nodes[node]["marked"] == 1:
            neighbors =  G.neighbors(node)
            i = 0
            for node_1, node_2 in combinations(neighbors, 2):
                G.add_edge(node_1, node_2)
                i += 1
                if i > 5:
                    break
            to_remove.append(node)

        if G.nodes[node]["marked"] == 0:
            attrs[node] = G.nodes[node]

    for node in to_remove:
        G.remove_node(node)

    attrs = get_attrs(G, False, attrs)
    nx.set_node_attributes(G, attrs)
    
    try:
        centralities = nx.betweenness_centrality(G, k=10, seed=0, weight='friends_bought')
    except:
        return None

    activation_list = list(G.nodes)
    activation_list.sort(key=lambda x: centralities[x], reverse=True)
    return activation_list

def free_spread(G: nx.Graph, current_iter: int, current_budget: float):
    # сколько прибыли мы получим, если просто дадим тренду распространиться без заражения
    G = G.copy()
    max_budget = current_budget

    for step in range(current_iter, 180):
        step_price = 0
        step_reward = 0
        step_users = 0
        update_today = []

        for n in G.nodes():  # проходим по всем вершинам в графе
            fb = 0
            for nei in [n for n in G.neighbors(n)]:  # подсчитываем зараженных соседей
                if G.nodes[nei]["marked"] == 1:
                    fb += 1
            G.nodes[n]["friends_bought"] = fb

            if G.nodes[n]["friends_bought"] * 5 >= G.nodes[n]["friends"] and G.nodes[n]["marked"] == 0:
                G.nodes[n]["marked"] = 2
                update_today.append(n)
                step_users += 1
                step_reward += REWARD

        for n in update_today:
            G.nodes[n]["marked"] = 1

        current_budget = current_budget - step_price + step_reward
        if current_budget > max_budget:
            max_budget = current_budget
        elif current_budget == max_budget:
            break
    
    return max_budget