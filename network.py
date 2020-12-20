import csv
import json
import re
import itertools
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import snap
import numpy

# setting up data structures to map actor IDs to objects in order to increase run time.
csv.field_size_limit(100000000)
curr_actor_id = 1
all_actors = dict()
all_actors_id_map = dict()
all_actors_frequencies = dict()
edges = set()
weights = dict()
movies = list()
movies_dict = dict()
edges_last_60_20 = set()
comm = list()
PG = nx.Graph()

class Actor:

    def __init__(self, name: str, id:int):
        self.filmography = set()
        self.name = name
        self.id = id
    def getFilms(self):
        return self.filmography

    def getName(self):
        return self.name

    def getId(self):
        return self.id

    def updateFilms(self, film:int):
        self.filmography.add(film)


class Movie:

    def __init__(self, id: int):
        self.actors = set()
        self.name = ""
        self.id = id
        self.year = 0

    def getName(self):
        return self.name

    def getActors(self):
        return self.actors

    def getId(self):
        return self.id

    def getDate(self):
        return self.year

    def updateActors(self, actor:Actor):
        self.actors.add(actor)

    def updateActors(self, actors_to_add:set()):
        for x in actors_to_add:
            self.actors.add(x)

    def setDate(self, i: int):
        self.year = i

#parsing data from csv and dropping crew column
reader = pd.read_csv('credits.csv', header = 0)
crewless = reader.drop('crew', axis = 1)
cleanup = re.compile('[^a-zA-Z\s]')

#skip the header row
row = crewless.iterrows()

#loop through each row
for x in range(len(reader.index)):
    cur_row = next(row)
    data = cur_row[1][0]
    id = cur_row[1][1]
    actors = set()

    #create an instance of a Movie for each row
    movie = Movie(int(id))
    movies.append(movie)
    movies_dict[id] = movie

    #split the string around each name
    split_around_names = data.split('name')

    #parse actors, and create an instance of Actor for each actor in each movie
    for y in range(1, len(split_around_names)):
        #Cleaning up characters and spaces around the actor's name
        actorName = str(split_around_names[y].split('order')[0])
        actorName = cleanup.sub(' ', actorName)
        actorName = actorName.strip()
        #Create the Actor and update his/her filmography
        if actorName not in all_actors.keys():
            a = Actor(actorName, curr_actor_id)
            curr_actor_id += 1
            a.updateFilms(movie)
            actors.add(a)
            all_actors[actorName] = a
            all_actors_frequencies[a] = 1
            all_actors_id_map[curr_actor_id] = a
        else:
            all_actors[actorName].updateFilms(movie)
            all_actors_frequencies[a] += 1
            actors.add(all_actors[actorName])
    #Update the set of actors per movie
    movie.updateActors(actors)

reader = pd.read_csv('movies_metadata.csv', header = 0)
reader.drop(reader.columns.difference(['id', 'release_date']), 1, inplace=True)
row = reader.iterrows()

cleaned_actors = set()
cleaned_movies_1 = set()
cleaned_movies = set()

# adding ids to movies from movie files
for x in range(len(reader.index)):
    cur_row = next(row)
    id = cur_row[1][0]
    date = cur_row[1][1]
    id = int(id)
    year = date[:4]
    year_int = int(year)
    if id in movies_dict.keys():
        movies_dict[id].setDate(year_int)
        cleaned_movies_1.add(movies_dict[id])


def clean(threshold: int):
    for actorName in all_actors.keys():
        if len(all_actors[actorName].getFilms()) > threshold:
            cleaned_actors.add(all_actors[actorName])
        else:
            for movie in all_actors[actorName].getFilms():
                if all_actors[actorName] in movie.getActors():
                    movie.getActors().remove(all_actors[actorName])


def clean_movies(threshold: int):
    for movie in cleaned_movies_1:
        if 2017 - movie.getDate() <= threshold:
            cleaned_movies.add(movie)
        else:
            for actor in movie.getActors():
                s = actor.getFilms()
                s.remove(movie)


def createGraph():
    counter = 0
    G = nx.Graph()
    PG_actors = set()

    #fill graph with nodes
    for actor in cleaned_actors:
        G.add_node(actor.getId())

    #generate a list of edges and weights based on frequencie of combination appearances
    for movie in cleaned_movies:
        actorIds = set()
        for actor in movie.getActors():
            actorIds.add(actor.getId())
        combinations = itertools.combinations(actorIds, 2)
        for comb in combinations:
            reverse = comb[::-1]
            if (comb not in edges) and (reverse not in edges):
                counter+=1
                if (2017 - movie.getDate() < 60 and 2017 - movie.getDate() > 20):
                    if (comb not in edges_last_60_20) and (reverse not in edges_last_60_20):
                        edges_last_60_20.add(comb)
                edges.add(comb)
                weights[comb] = 1
            else:
                if comb in edges:
                    weights[comb] = weights[comb] + 1
                elif reverse in edges:
                    weights[reverse] = weights[reverse] + 1
    G.add_edges_from(edges)
    for x in edges_last_60_20:
        if x[0] not in PG_actors:
            PG_actors.add(x[0])
        if x[1] not in PG_actors:
            PG_actors.add(x[1])
    PG.add_nodes_from(PG_actors)
    PG.add_edges_from(edges_last_60_20)
    return G


def centrality_analysis():
    types = [nx.eigenvector_centrality, nx.harmonic_centrality, nx.degree_centrality]

    for x in types:

        # based upon cleaning values chosen, choose a directory to store results to.
        file = open('./centrality/40_10/centrality_results_'+x.__name__+'.txt', 'w')
        nodes = x(graph)
        top_10 = list()
        top_10_ids = list()

        sorted_values = list(nodes.values())
        sorted_values.sort()
        sorted_values.reverse()

        top_10 = sorted_values[0]
        # print(sorted_values)

        # for y in top_10:
        for x in nodes.keys():
            if nodes[x] == top_10:
                top_10_ids.append(x)

        file.write(str(len(top_10_ids)) + '\n')
        for x in top_10_ids:
            for y in cleaned_actors:
                if x == y.getId():
                    print(y.getName())
                    #file.write(y.getName() + '\n')
        file.close()


def community_analysis():
    f = open('./community/communities_outputs.txt', 'w')
    communities_generator = nx.community.girvan_newman(graph)
    communities = next(communities_generator)
    size = len(communities)
    while size < 10:
        print(communities)
        communities = next(communities_generator)
        size = len(communities)
        f.write('community iteration: size = {}, {} \n'.format(size, communities))


def link_pred():
    splPG = dict(nx.all_pairs_shortest_path_length(PG, cutoff=2))
    friends_PG = list()
    for x in splPG.keys():
        for y in splPG[x].keys():
            if splPG[x][y] == 2:
                l = list()
                l.append(x)
                l.append(y)
                friends_PG.append(l)
    predictions = nx.jaccard_coefficient(PG, friends_PG)
    results = list()
    for x in predictions:
        results.append(x)
    results.sort(key=lambda x: x[2])
    results.reverse()

    k_vals = [10,20,50,100]
    for k in k_vals:
        f = open('./link_pred/link_prediction_values_jaccard' + str(k) + '.txt', 'w')
        count = 0
        while (count < k):
             print('({}, {}),jaccard: {}'.format(all_actors_id_map[results[count][0]].getName(), all_actors_id_map[results[count][1]].getName(), results[count][2]))
             f.write('({}, {}),jaccard: {}\n'.format(all_actors_id_map[results[count][0]].getName(),all_actors_id_map[results[count][1]].getName(),results[count][2]))
             count+=1
        top_k = list()
        precision_at_k = 0
        for x in range(k):
            top_k.append(results[x])
        count = 0
        for val in top_k:
            tup = (val[0], val[1])
            if tup in edges:
                count += 1
        precision_at_k = count / k
        print('precision @ K{}: {}\n'.format(k, precision_at_k))
        f.write('precision @ K{}: {}'.format(k, precision_at_k))
        f.close()

#Convert community results from IDs to Actor name
def convert_id_actor():
    file = open('./community_/communities_outputs.txt')
    for row in file:
        items = row.split(', ')
        i = 0
        while i < len(items):
            items[i].strip('\n')
            items[i] = int(items[i])
            i+=1
        i = 0
        this_row = list()
        i= 0
        while i < len(items):
            this_row.append(items[i])
            i+=1
        comm.append(this_row)
    file.close()
    file = open('./actorname_communities.txt', 'w')
    for x in range(len(comm)):
        for y in range(len(comm[x])):
            try:
                comm[x][y] = all_actors_id_map[comm[x][y]].getName()
            except:
                comm[x][y] = 'None'
    comm.reverse()
    for x in range(len(comm)):
        print("Community #{}: {}".format(x, comm[x]))
        file.write("Community #{}: {}\n".format(x, comm[x]))
        file.flush()
    file.close()


clean_movies(60)
clean(30)

graph = createGraph()
print(nx.info(graph))
print(nx.info(PG))


# To perform the analysis, uncomment the respective function(s); additionally, uncomment  #convert_id_actor() for community_analysis.
# centrality_analysis()
# community_analysis()
# convert_id_actor()
# link_pred()
