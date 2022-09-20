import time
import matplotlib.pyplot as plt
import ocpa.algo.filtering.log.time_filtering
from ocpa.objects.log.obj import OCEL
from ocpa.algo.discovery.ocpn import algorithm as ocpn_discovery_factory
import pandas as pd
import ocpa.algo.filtering.log.trace_filtering as trace_filtering
import ocpa.algo.evaluation.precision_and_fitness.utils as evaluation_utils
import ocpa.algo.evaluation.precision_and_fitness.evaluator as precision_fitness_evaluator
import ocpa.visualization.oc_petri_net.factory as vis_factory
import ocpa.visualization.log.variants.factory as log_viz
import ocpa.objects.log.importer.ocel.factory as import_factory
import ocpa.algo.feature_extraction.factory as feature_extraction
from ocpa.algo.feature_extraction import time_series
from ocpa.algo.feature_extraction import tabular, sequential
import numpy as np
from ast import literal_eval
from gnn_utils import *
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score
from statistics import median as median
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import keras.backend as K
import shap
import random
from datetime import timedelta





def avg(x):
    if len(x) == 0:
        return np.nan
    return sum(x)/len(x)

def std_dev(x):
    m = sum(x) / len(x)
    return sum((xi - m) ** 2 for xi in x) / len(x)

filename = "example_logs/mdl/BPI2017-Final.csv"
ots = ["application", "offer"]


event_df = pd.read_csv(filename, sep=',')#[:2000]
event_df["event_timestamp"] = pd.to_datetime(event_df["event_timestamp"])
event_df = event_df.sort_values(by='event_timestamp')

def eval(x):
    try:
        return literal_eval(x.replace('set()', '{}'))
    except:
        return []



print(event_df)
for ot in ots:
    event_df[ot] = event_df[ot].apply(eval)
event_df["event_id"] = list(range(0, len(event_df)))
event_df.index = list(range(0, len(event_df)))
event_df["event_id"] = event_df["event_id"].astype(float).astype(int)
event_df["event_start_timestamp"] = pd.to_datetime(event_df["event_start_timestamp"])
#####FAKE FEATURE VALUE
event_df["event_fake_feat"] = 1
ocel = OCEL(event_df, ots)
t_start = time.time()
print("Number of process executions: "+str(len(ocel.cases)))
print(str(time.time()-t_start))
print(ocel.log)
activities = list(set(ocel.log["event_activity"].tolist()))
print(str(len(activities))+" actvities")
F = [(feature_extraction.EVENT_REMAINING_TIME,()),
     (feature_extraction.EVENT_PREVIOUS_TYPE_COUNT,("offer",)),
     (feature_extraction.EVENT_ELAPSED_TIME,())] + [(feature_extraction.EVENT_AGG_PREVIOUS_CHAR_VALUES,("event_RequestedAmount",max))] \
    + [(feature_extraction.EVENT_PRECEDING_ACTIVITES,(act,)) for act in activities]

feature_storage = feature_extraction.apply(ocel, F, [])
feature_storage.extract_normalized_train_test_split(0.3,state = 3)


# keep list of first three events for comparability of regression use case
events_to_remove = []
for g in tqdm(feature_storage.feature_graphs):
    event_ids = [n.event_id for n in g.nodes]
    event_ids.sort()
    events_to_remove = events_to_remove + event_ids[:3]

label_order = None

accuracy_dict = {}

#CASE STUDY 1 - VISUALIZING TABLE
if True:
    print("___________________________")
    print("USE CASE 1 - Time series visualization")
    print("___________________________")
    feat_to_s = {}
    f_in = ocpa.algo.filtering.log.time_filtering.events
    s_time= time.time()
    s, time_index = time_series.construct_time_series(ocel, timedelta(days=7),
                                                      [(avg, (feature_extraction.EVENT_TYPE_COUNT,("offer",))),
                                                       (avg, (feature_extraction.EVENT_CHAR_VALUE,("event_RequestedAmount",)))],
                                                      [],
                                                      f_in)
    print("total time series: " + str(time.time() - s_time))
    for feat in s.keys():
        if feat not in feat_to_s.keys():
            feat_to_s[feat] = []
        feat_to_s[feat].append((s[feat], time_index))
    print(feat_to_s)
    data_df = pd.DataFrame({"date":feat_to_s[list(feat_to_s.keys())[0]][0][1]})
    for feat in feat_to_s.keys():
        name = ""
        if feat == list(feat_to_s.keys())[0]:
            name = "Average Number of Objects"
        else:
            name = "Average Remaining Time in s"
        data_df[name] = [feat_to_s[feat][0][0][i] for i in range(0,len(feat_to_s[feat][0][0]))]
    plt.clf()
    plt.rcParams["axes.labelsize"] = 32
    plt.rcParams["axes.titlesize"] = 32
    plt.figure(figsize=(8, 3))
    sns.set(rc={'figure.figsize': (12, 6)})
    data_df.set_index('date', inplace=True)
    sns.set_style("darkgrid")
    plot_ = sns.lineplot(data=data_df["Average Number of Objects"], color = "#4C72B0", linewidth = 2)
    ax2 = plt.twinx()
    sns.lineplot(data=data_df["Average Remaining Time in s"], ax = ax2, color="#DD8452", linewidth = 2)
    for index, label in enumerate(plot_.get_xticklabels()):
        if index % 2 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    #plot_.legend()
    plot_.set_title("Time Series",fontsize=14)
    plot_.set_ylabel("Average Number of Offer Objects", color = "#4C72B0",fontsize=11)
    #plot_.set_yticklabels(plot_.get_yticks(), size=14)
    ax2.set_ylabel("Average Requested Amount", color="#DD8452",fontsize=11)
    #ax2.set_yticklabels(ax2.get_yticks(), size=14)
    plot_.set_xlabel(
        "Date",fontsize=14)
    plt.tight_layout()
    plt.savefig("Time_series.png",dpi=600)
##Case Study 2 - Regression
if True:
    print("___________________________")
    print("USE CASE 2 - Regression prediction")
    print("___________________________")
    train_table = tabular.construct_table(feature_storage, index_list = feature_storage.training_indices, exclude_events = events_to_remove)
    test_table = tabular.construct_table(feature_storage, index_list = feature_storage.test_indices, exclude_events = events_to_remove)


    y_train, y_test = train_table[F[0]], test_table[F[0]]
    x_train, x_test = train_table.drop(F[0], axis = 1), test_table.drop(F[0], axis = 1)
    #y.rename(columns={f: ''.join(f) for f in y.columns}, inplace=True)


    #x_train.rename(columns={f:str(f) for f in x_train.columns}, inplace=True)
    mapping_names = {feature_extraction.EVENT_PRECEDING_ACTIVITES: "Prec. activities:",
                     feature_extraction.EVENT_ELAPSED_TIME: "Elapsed time",
                     feature_extraction.EVENT_PREVIOUS_TYPE_COUNT: "Previous objects of:",
                     feature_extraction.EVENT_AGG_PREVIOUS_CHAR_VALUES: "Max prev.:",
                     feature_extraction.EVENT_REMAINING_TIME: "Remaining time"}
    renaming_dict = dict()
    for f in x_train.columns:
        if len(f[1]) != 0:
            renaming_dict[f] = mapping_names[f[0]] + " " + f[1][0]
        else:
            renaming_dict[f] = mapping_names[f[0]]
    x_train.rename(columns=renaming_dict, inplace=True)
    x_test.rename(columns=renaming_dict, inplace=True)
    #print(x_train.columns)
    X100 = shap.utils.sample(x_train, 500)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    avg_rem = avg(y_train)
    print('MAE baseline: ', mean_absolute_error(y_test, [avg_rem for elem in y_test]))
    print('MAE: ', mean_absolute_error(y_test, y_pred))
    #print(y_test)
    #print(y_pred)
    #test = pd.DataFrame({'Predicted value': y_pred, 'Actual value': y_test})
    fig = plt.figure(figsize=(10, 8))
    #test = test.reset_index()
    #test = test.drop(['index'], axis=1)
    plt.clf()
    accuracy_dict['regression'] = {
        'train_MAE': mean_absolute_error(y_train, model.predict(x_train)),
        'test_MAE': mean_absolute_error(y_test, y_pred)
    }
    print(pd.DataFrame(accuracy_dict))
    if True:
        explainer = shap.Explainer(model.predict, x_train,feature_names=x_train.columns.tolist())
        shap_values = explainer(x_test[:1000])
        #shap.plots.beeswarm(shap_values,max_display=40, show = False)
        shap.summary_plot(shap_values, max_display=60, show=False)
        plt.gcf().axes[-1].set_aspect(100)
        plt.gcf().axes[-1].set_box_aspect(100)
        locs, labels = plt.yticks()
        label_order = [l.get_text() for l in labels][::-1]
        plt.savefig('shap_regression_bee.png',bbox_inches='tight')

#CASE Study 3 - Visualizing Sequence
if True:
    print("___________________________")
    print("USE CASE 3 - Sequence based variant visualization")
    print("___________________________")
    #activities = list(set(ocel.log["event_activity"].tolist()))
    F3 = [(feature_extraction.EVENT_ACTIVITY,(act,)) for act in activities] + [(feature_extraction.EVENT_TYPE_COUNT,(ot,)) for ot in ots]
    feature_storage3 = feature_extraction.apply(ocel,F3,[])
    sequences = sequential.construct_sequence(feature_storage3)
    for v_id in [61]:
        print(v_id)
        c_id = ocel.variants_dict[ocel.variants[v_id]][0]
        print(sequences[c_id])
        print(len(sequences[c_id]))
        print(len(ocel.cases[c_id]))
        print(ocel.variant_frequency[v_id])

#CASE Study 4 - Prediction - LSTM
if True:
    print("___________________________")
    print("USE CASE 4 - LSTM prediction")
    print("___________________________")
    k=4
    features = [feat for feat in F if feat != (feature_extraction.EVENT_REMAINING_TIME,())]
    target = (feature_extraction.EVENT_REMAINING_TIME,())
    train_sequences = sequential.construct_sequence(feature_storage, index_list=feature_storage.training_indices)
    test_sequences = sequential.construct_sequence(feature_storage, index_list=feature_storage.test_indices)

    x_train, y_train = sequential.construct_k_dataset(train_sequences,k,features,target)
    x_test, y_test = sequential.construct_k_dataset(test_sequences, k, features, target)
    x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
    X100 = shap.utils.sample(x_train, 100)
    regressor = Sequential()

    regressor.add(LSTM(units=10, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=10))
    regressor.add(Dropout(0.1))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics = 'mae')
    K.set_value(regressor.optimizer.learning_rate, 0.005)
    best_weights_callback = tf.keras.callbacks.ModelCheckpoint('lstm_checkpoint.h5', monitor = 'val_loss', save_best_only = True, verbose = 1)

    history = regressor.fit(x_train, y_train, validation_split = 0.2, epochs=30, batch_size=64, callbacks = best_weights_callback)
    regressor.load_weights('lstm_checkpoint.h5')

    y_pred = regressor.predict(x_test)
    y_pred = np.transpose(y_pred)
    y_pred = y_pred[0]
    #print(y_pred[:10])
    #print(y_test[:10])
    avg_rem = avg(y_train)
    print('MAE baseline: ', mean_absolute_error(y_test, [avg_rem for elem in y_test]))
    print('MAE: ', mean_absolute_error(y_test, y_pred))
    #test = pd.DataFrame({'Predicted value': y_pred, 'Actual value': y_test})
    accuracy_dict['lstm'] = {
        'train_MAE': history.history['mae'][np.argmin(history.history['val_loss'])],
        'val_MAE': history.history['val_mae'][np.argmin(history.history['val_loss'])],
        'test_MAE': mean_absolute_error(y_test, y_pred)
    }


    print(pd.DataFrame(accuracy_dict))
    fig = plt.figure(figsize=(7, 6))
    #test = test.reset_index()
    #test = test.drop(['index'], axis=1)
    #plt.plot(test[:100])
    #plt.legend(['Actual value', 'Predicted value'])
    #plt.savefig("prediction_lstm.png", dpi=600)
    #plt.clf()

    def f(X):
        X = X.reshape(X.shape[0], k,int(X.shape[1]/k))
        return regressor.predict(X)
    #print(X100.shape)
    #print(x_train.shape)
    number_shap = 40
    explainer = shap.KernelExplainer(f, X100.reshape(X100.shape[0],X100.shape[1]*X100.shape[2]))#, feature_names=[str(f) for f in features])
    shap_values = explainer.shap_values(x_train[:number_shap].reshape(x_train[:number_shap].shape[0],x_train[3].shape[0]*x_train[3].shape[1]), nsamples = 100)
    shap_values = shap_values[0].reshape(shap_values[0].shape[0],k,int(shap_values[0].shape[1]/k))
    print(shap_values[0])
    mapping_names = {feature_extraction.EVENT_PRECEDING_ACTIVITES: "Prec. activities:",
                     feature_extraction.EVENT_ELAPSED_TIME: "Elapsed time",
                     feature_extraction.EVENT_PREVIOUS_TYPE_COUNT: "Previous objects of:",
                     feature_extraction.EVENT_AGG_PREVIOUS_CHAR_VALUES: "Max prev.:",
                     feature_extraction.EVENT_REMAINING_TIME: "Remaining time"}
    renaming_dict = dict()
    for f in features:
        if len(f[1]) != 0:
            renaming_dict[f] = mapping_names[f[0]] + " " + f[1][0]
        else:
            renaming_dict[f] = mapping_names[f[0]]
    for i in range(0,len(shap_values)):
        plt.clf()
        #first, reorder the features such that their order is the same as the beeswarm plot
        curr_order = [renaming_dict[f] for f in features]
        mapping_from_curr_to_new = {}
        for c in curr_order:
            for i_n in range(0,len(label_order)):
                n= label_order[i_n]
                if c == n:
                    mapping_from_curr_to_new[c] = i_n
        #reorder shap values
        new_shap = np.copy(shap_values[i])
        for i_c in range(0,len(curr_order)):
            feat = curr_order[i_c]
            for i_f in range(0,k):
                new_shap[i_f][mapping_from_curr_to_new[feat]] = shap_values[i][i_f][i_c]


        ax = sns.heatmap(np.transpose(new_shap),xticklabels = ["-3 events","-2 events","-1 event","current event"], yticklabels = label_order)
        plt.xticks(rotation=45)
        ax.figure.tight_layout()
        #print(shap_values)
        #shap.plots.waterfall(shap_values[0][0], max_display=160, show=False)
        plt.savefig('shap_lstm_instance_'+str(i)+'.png', dpi=600)




#Case study 5 - Graph-based variant visualization
if True:
    print("___________________________")
    print("USE CASE 5 - Graph-based variant visualization")
    print("___________________________")
    layouting = log_viz.apply(ocel)
    print(layouting[ocel.variants[61]])



#Case study 6 - Graph Neural Network Prediction
if True:
    print("___________________________")
    print("USE CASE 6 - Graph neural network prediction")
    print("___________________________")
    train_idx, val_idx = train_test_split(feature_storage.training_indices, test_size = 0.2)
    x_train, y_train = generate_graph_dataset(feature_storage.feature_graphs, train_idx, ocel)
    x_val, y_val = generate_graph_dataset(feature_storage.feature_graphs, val_idx, ocel)
    x_test, y_test = generate_graph_dataset(feature_storage.feature_graphs, feature_storage.test_indices, ocel)

    # initialize data loaders
    train_loader = GraphDataLoader(
        x_train,
        y_train,
        batch_size = 64,
        shuffle = True,
        add_self_loop = True,
        make_bidirected = False,
        on_gpu = False
    )
    val_loader = GraphDataLoader(
        x_val,
        y_val,
        batch_size = 64,
        shuffle = True,
        add_self_loop = True,
        make_bidirected = False,
        on_gpu = False
    )
    test_loader = GraphDataLoader(
        x_test,
        y_test,
        batch_size = 128,
        shuffle = False,
        add_self_loop = True,
        make_bidirected = False,
        on_gpu = False
    )

    # define GCN model
    tf.keras.backend.clear_session()
    model = GCN(24, 24)
    optimizer = tf.keras.optimizers.Adam(lr = 0.01)
    loss_function = tf.keras.losses.MeanAbsoluteError()

    # run tensorflow training loop
    epochs = 30
    iter_idx = np.arange(0, train_loader.__len__())
    loss_history = []
    val_loss_history = []
    step_losses = []
    for e in range(epochs):
        print('Running epoch:', e)
        np.random.shuffle(iter_idx)
        current_loss = step = 0
        for batch_id in tqdm(iter_idx):
            step += 1
            dgl_batch, label_batch = train_loader.__getitem__(batch_id)
            with tf.GradientTape() as tape:
                pred = model(dgl_batch, dgl_batch.ndata['features'])
                loss = loss_function(label_batch, pred)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            step_losses.append(loss.numpy())
            current_loss += loss.numpy()
            # if (step % 100 == 0): print('Loss: %s'%((current_loss / step)))
            loss_history.append(current_loss / step)
        val_predictions, val_labels = evaluate_gnn(val_loader, model)
        val_loss = tf.keras.metrics.mean_absolute_error(np.squeeze(val_labels), np.squeeze(val_predictions)).numpy()
        print('    Validation MAE GNN:', val_loss)
        if len(val_loss_history) < 1:
            model.save_weights('gnn_checkpoint.tf')
            print('    GNN checkpoint saved.')
        else:
            if val_loss < np.min(val_loss_history):
                model.save_weights('gnn_checkpoint.tf')
                print('    GNN checkpoint saved.')
        val_loss_history.append(val_loss)

    # visualize training progress
    pd.DataFrame({'loss': loss_history, 'step_losses': step_losses}).plot(subplots = True, layout = (1, 2), sharey = True)

    # restore weights from best epoch
    cp_status = model.load_weights('gnn_checkpoint.tf')
    cp_status.assert_consumed()

    # generate predictions and calculate MAE for train, val & test sets
    train_predictions, train_labels = evaluate_gnn(train_loader, model)
    val_predictions, val_labels = evaluate_gnn(val_loader, model)
    test_predictions, test_labels = evaluate_gnn(test_loader, model)
    mean_prediction = np.mean(np.array(y_train))
    print('MAE baseline: ')
    print(mean_absolute_error(test_labels, np.repeat(mean_prediction, len(test_labels))))
    print('MAE GNN: ')
    print(mean_absolute_error(test_predictions, test_labels))

    # record performance of GNN
    accuracy_dict['gnn'] = {
        'train_MAE': mean_absolute_error(train_predictions, train_labels),
        'val_MAE': mean_absolute_error(val_predictions, val_labels),
        'test_MAE': mean_absolute_error(test_predictions, test_labels)
    }
    print(pd.DataFrame(accuracy_dict))
    # calculate shap values for the presence of edges for sample instance
    test_graph = x_test[2]
    #visualize_instance(test_graph, y_test[2])
    test_graph = dgl.add_self_loop(test_graph)
    test_features = test_graph.ndata['features']
    test_features = test_features.numpy()
    test_features.shape

    # define prediction function
    def f(edge_selection):

        all_preds = []

        for i in edge_selection:
            idx = np.concatenate([i, np.array([1, 1, 1, 1])], axis = 0).astype('bool')
            edges = test_graph.edges()
            selected_from = edges[0].numpy()[idx]
            selected_to = edges[1].numpy()[idx]
            new_graph = dgl.graph(
                data = (selected_from, selected_to)
            )
            new_graph.ndata['features'] = test_graph.ndata['features']
            new_graph.ndata['remaining_time'] = test_graph.ndata['remaining_time']
            new_graph.ndata['event_indices'] = test_graph.ndata['event_indices']

            with tf.device('CPU:0'):
                pred = model(new_graph, new_graph.ndata['features']).numpy().squeeze()

            all_preds.append(pred)
        all_preds = np.array(all_preds)

        return all_preds

    # explain instance
    explainer = shap.KernelExplainer(f, np.zeros((1, 4)))
    shap_values = explainer.shap_values(np.ones((1, 4)), nsamples = 1000)
    shap_values

    plt.clf()
    # visualize instance
    nx_G = x_test[2].cpu().to_networkx(node_attrs = ['remaining_time', 'event_indices'])
    pos = nx.kamada_kawai_layout(nx_G)
    edges = [i for i in nx_G.edges() if (i[0] != i[1])]
    edge_labels = {k: v for k, v in zip(edges, np.round(shap_values[0], 2))}
    nx.draw(nx_G, pos, with_labels = True, node_color=[[.7, .7, .7]], font_size = 10)
    nx.draw_networkx_edge_labels(nx_G, pos, edge_labels = edge_labels)
    plt.savefig('shap_graph.png')
    pd.DataFrame(accuracy_dict).to_csv('results_table.csv')
