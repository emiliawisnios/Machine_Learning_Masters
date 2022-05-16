"""Practical 5"""
import sqlite3

import numpy as np

from .nlp import normalize

# loading databases
domains = ['restaurant']
dbs = {}
for domain in domains:
    db = 'db/{}-dbase.db'.format(domain)
    conn = sqlite3.connect(db)
    c = conn.cursor()
    dbs[domain] = c


# TODO TASK b)
# Create a one-hot encoding informing how many entities are available to 
# the user. The output list should consists of 6 buckets.
def one_hot_vector(num_entities, domain, vector):
    """Return number of available entities for particular domain.
    
    Args:
        num_entites: int
        domain: str
        vector: np.array

    Returns:
        available_entities: np.array
    """
    available_entities = np.array([0, 0, 0, 0, 0, 0])
    # YOUR CODE HERE
    buckets = [0, 0, 2, 5, 10, 40, np.inf]
    for i in range(len(buckets) - 1):
        if buckets[i] <= num_entities < buckets[i + 1]:
            available_entities[i] = 1
        else:
            available_entities[i] = 0
    # YOUR CODE ENDS HERE
    return available_entities


def queryResult(domain, turn):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""
    # query the db
    sql_query = "select * from {}".format(domain)

    flag = True
    for key, val in list(turn['metadata'][domain]['semi'].items()):
        if (key == 'requested' or val == "" or val == "dont care" or
                val == 'not mentioned' or val == "don't care" or val == "dontcare"
                or val == "do n't care"):
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                if key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                if key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    num_entities = len(dbs[domain].execute(sql_query).fetchall())

    return num_entities


def query_result_venues(domain, turn, real_belief=False):
    sql_query = "select * from {}".format(domain)

    if real_belief == True:
        items = list(turn.items())
    else:
        items = list(turn['metadata'][domain]['semi'].items())

    flag = True
    for key, val in items:
        if (key == "requested" or val == "" or val == "dontcare" or
                val == 'not mentioned' or val == "don't care" or
                val == "dont care" or val == "do n't care"):
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                if key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                if key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    try:
        return dbs[domain].execute(sql_query).fetchall()
    except:
        return []
