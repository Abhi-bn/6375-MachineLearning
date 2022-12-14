
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from MN import MN
from BTP import BTP
import time
import queue
import threading


class Data:
    def __init__(self, fpath):

        f = open(fpath, "r")

        self.nvars = int(f.readline())  # 1

        line = np.asarray(f.readline().split(), dtype=np.int32)  # 2
        self.evid_var_ids = line[1:]
        evid_indices = range(1, self.evid_var_ids.shape[0]*2, 2)

        line = np.asarray(f.readline().split(), dtype=np.int32)  # 3
        self.query_var_ids = line[1:]
        query_indices = range(
            self.evid_var_ids.shape[0]*2+1, (self.evid_var_ids.shape[0]+self.query_var_ids.shape[0])*2, 2)

        line = np.asarray(f.readline().split(), dtype=np.int32)  # 4
        self.hidden_var_ids = line[1:]

        line = f.readline()  # 5
        self.nproblems = int(f.readline())  # 6

        self.evid_assignments = []
        self.query_assignments = []
        self.weights = []
        for i in range(self.nproblems):
            line = np.asarray(f.readline().split(), dtype=float)
            self.evid_assignments.append(np.asarray(
                line[evid_indices], dtype=np.int32))
            self.query_assignments.append(np.asarray(
                line[query_indices], dtype=np.int32))
            self.weights.append(line[-1])
        self.evid_assignments = np.asarray(self.evid_assignments)
        self.query_assignments = np.asarray(self.query_assignments)
        self.weights = np.asarray(self.weights)
        self.hidden_assignments = []

    def convertToXYWithH(self, hidden_assignments):
        return (np.concatenate((self.evid_assignments, hidden_assignments), axis=1), self.query_assignments)

    def convertToXY(self):
        return (self.evid_assignments, self.query_assignments)

    def convertResults(self, query_predictions):
        out = np.zeros(
            (query_predictions.shape[0], 1+2*self.query_var_ids.shape[0]), dtype=int)
        out[:, 2::2] = query_predictions[:, :]
        out[:, 1::2] = self.query_var_ids
        out[:, 0] = self.query_var_ids.shape[0]
        return out

    def computeLogProb(self, dir_path, order, X, y):
        mn = MN()
        mn.read(dir_path+'.uai')
        out = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            for j in range(len(self.evid_var_ids)):
                mn.setEvidence(self.evid_var_ids[j], X[i][j])
            for j in range(y.shape[1]):
                mn.setEvidence(self.query_var_ids[j], y[i][j])
            btp = BTP(mn, order)
            out[i] = np.log10(btp.getPR())
        return out

    @staticmethod
    def computeErr(true_ll, pred_ll):
        return np.sum(true_ll)-np.sum(pred_ll)

    @staticmethod
    def computeScore(err, max_err):
        return np.max((0, 100*(1.0-err/max_err)))


data_directory = './content/MLC/'
dname = 'Sample_1_MLC_2022'
data = Data(data_directory+dname+'.data')


X, y = data.convertToXY()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# # Use this the first time to generate order
# mn_for_order = MN()
# mn_for_order.read(data_directory+dname+'.uai')
# temp = BTP(mn_for_order)
# temp.getOrder(2)


# # storing order, it takes hell lot of time to get Order
# np.savetxt(X=temp.order, delimiter=' ', fmt='%d',
#            fname=data_directory+dname+'.order')

# Loading stored Order here
load_order = np.loadtxt(data_directory+dname+'.order',
                        dtype=np.int32, delimiter=' ').astype(np.int32)


def generate_features(ev_id, ev_ass, q_id, q_ass):
    mn = MN()
    mn.read(data_directory+dname+'.uai')
    for j in range(len(ev_id)):
        mn.setEvidence(ev_id[j], ev_ass[j])
    for j in range(len(q_id)):
        mn.setEvidence(q_id[j], q_ass[j])

    btp = BTP(mn, load_order)
    btp.performUpwardPass()

    store_all = {}
    # storing reversed order only
    for i, bucket in enumerate(reversed(btp.buckets)):
        # don't care about empty bucket
        if len(bucket) == 0:
            continue
        for func in bucket:
            # loading in reversed order, since buckets are stored in order
            bucket_id = btp.order[len(btp.buckets) - i - 1]
            # can also get bucket_id from most id occurrence in that bucket (not concrete but i tried this first)
            # bucket_id = max(set(func.getVarIDs()), key=lambda x: list(func.getVarIDs()).count(x))
            store_all.setdefault(bucket_id, [])
            min_order = []
            # seeing all the id's
            for id in func.getVarIDs():
                min_order.append([list(btp.order).index(id), id])
            # irrespective of it contains bucket_id, we are placing at the minimum order
            # coz if not we will be missing an hidden assignment
            mini = min(min_order, key=lambda x: x[0])
            store_all.setdefault(mini[1], [])
            store_all[mini[1]].append(func)
    hidden_assignments = {}
    for key in reversed(btp.order):
        if store_all.get(key) == None:
            # its either query or evidence
            continue
        if len(store_all[key]) == 0:
            # IDK what to do if this comes up
            assert ("should not come here")
        # print(key)
        max_val_0 = []
        max_val_1 = []
        for func in store_all[key]:
            if len(func.getVarIDs()) > 1:
                solved_func = func.instantiateEvid()
                max_val_0.append(solved_func.getPotential()[0])
                max_val_1.append(solved_func.getPotential()[1])
            else:
                max_val_0.append(func.getPotential()[0])
                max_val_1.append(func.getPotential()[1])

        m0 = max(max_val_0)
        m1 = max(max_val_1)

        hidden_assignments[key] = 1 if m1 > m0 else 0
        mn.setEvidence(key, hidden_assignments[key])

    new_features = [0] * len(hidden_assignments)
    for key in hidden_assignments:
        new_features[list(data.hidden_var_ids).index(key)
                     ] = hidden_assignments[key]
    return new_features


data_set = np.zeros((data.nproblems, len(
    data.hidden_var_ids)))
start = time.time()
for index in range(data.evid_assignments.shape[0]):
    each = time.time()
    hidden_assignments = generate_features(
        data.evid_var_ids, data.evid_assignments[index], data.query_var_ids, data.query_assignments[index])

    data_set[index][:] = hidden_assignments
    if index % 500 == 0:
        np.savetxt(X=data_set, delimiter=' ', fmt='%d',
                   fname=data_directory+dname+'.new_features')
    print(index, "Done in ", time.time() - each)

# threads_to_start = 6  # or choose how many you want
# process_fast_queue = queue.Queue()


# def worker():
#     while True:
#         if process_fast_queue.empty() == True:
#             break
#         else:
#             st = time.time()
#             index = process_fast_queue.get()
#             hidden_assignments = generate_features(
#                 data.evid_var_ids, data.evid_assignments[index], data.query_var_ids, data.query_assignments[index])
#             data_set[index][:] = np.concatenate(
#                 [data.evid_assignments[index], hidden_assignments])

#             print(index, "Done in ", time.time() - st)
#             process_fast_queue.task_done()
#     return


# for i in range(data.evid_assignments.shape[0]):
#     process_fast_queue.put(i)


# for i in range(threads_to_start):
#     # daemon means that all threads will exit when the main thread exits
#     t = threading.Thread(target=worker, daemon=True)
#     t.start()


# process_fast_queue.join()
np.savetxt(X=data_set, delimiter=' ', fmt='%d',
           fname=data_directory+dname+'.new_features')
print("Total Time: ", time.time() - start)
