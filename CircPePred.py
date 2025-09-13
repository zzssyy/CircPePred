import data_processing
from sklearn.model_selection import KFold
import numpy as np
import ensemble_learning
import test_scores as score
import random
from sklearn import preprocessing

def CircPePred():
    
    sORFs_posi_file = './dataset3/dataset/sORFs-training.txt'
    sORFs_nega_file = './dataset3/dataset/non-sORFs-training.txt'
    posi_aas_file = './dataset3/dataset/peptides-training.txt'
    nega_aas_file = './dataset3/dataset/non-peptides-training.txt'
    posi_sorfs_samples, nega_sorfs_samples, sorfs_f_index, sorfs_f_name = data_processing.get_sorfs_dataset(sORFs_posi_file, sORFs_nega_file)
    posi_aas_file, nega_aas_file = data_processing.get_aas(sORFs_posi_file, sORFs_nega_file, posi_aas_file, nega_aas_file)
    posi_aas_samples, nega_aas_samples, aas_f_index, aas_f_name = data_processing.get_aas_dataset(posi_aas_file, nega_aas_file)
    

    posi_sorfs_samples, nega_sorfs_samples, posi_aas_samples, nega_aas_samples = data_processing.conn_shuf_split_dataset(posi_sorfs_samples, nega_sorfs_samples, posi_aas_samples, nega_aas_samples)
    
    
    print("starting sorfs feature selction...")
    posi_sorfs_samples, nega_sorfs_samples, sorfs_Eachfeature = data_processing.feature_selection(posi_sorfs_samples, nega_sorfs_samples, sorfs_f_index)
    
    print("starting aas feature selction...")
    posi_aas_samples, nega_aas_samples, aas_Eachfeature = data_processing.feature_selection(posi_aas_samples, nega_aas_samples, aas_f_index) 
    
    sorfs_fea_num = posi_sorfs_samples.shape[1]
    # posi_samples, nega_samples = posi_aas_samples, nega_aas_samples
    
    aas_f_nef = data_processing.get_fea_name_dict(aas_f_name, aas_Eachfeature)
    sorfs_f_nef = data_processing.get_fea_name_dict(sorfs_f_name, sorfs_Eachfeature)
    
    posi_samples, nega_samples = np.hstack((posi_sorfs_samples, posi_aas_samples)), np.hstack((nega_sorfs_samples, nega_aas_samples))
    metric_list = []
    n_fold = 5
    cluster_num = 5
    fold_index = 0
    for fold in range(n_fold):
        p_train = [i for i in range(posi_samples.shape[0]) if i%n_fold !=fold]
        p_test = [i for i in range(posi_samples.shape[0]) if i%n_fold ==fold]
        n_train = [i for i in range(nega_samples.shape[0]) if i%n_fold !=fold]
        n_test = [i for i in range(nega_samples.shape[0]) if i%n_fold ==fold]

        posi_train, posi_test = posi_samples[p_train,:], posi_samples[p_test,:]
        nega_train, nega_test = nega_samples[n_train,:], nega_samples[n_test,:]
        print(posi_train.shape, posi_test.shape)
        data_test = np.concatenate((posi_test[:,:-1], nega_test[:,:-1]), axis=0)
        y_test = np.concatenate((posi_test[:,-1], nega_test[:,-1]), axis=0)
        
        #split test data into aas and sorfs
        sorfs_data_test = data_test[:,:sorfs_fea_num]
        aas_data_test = data_test[:,sorfs_fea_num:]
                
        y_test_probs = []
        sub_factors = []
        nega_num = len(nega_train)
        cr_nega_train = nega_train
        for count in range(3):
            # splited_nega_trains = data_processing.spliting_by_clustering(cr_nega_train, cluster_num)
            splited_nega_trains = data_processing.spliting_by_clustering(cr_nega_train, cluster_num, method = 'PEDP')
            nega_train, cr_nega_train, nega_num= data_processing.sampling_from_clusters(splited_nega_trains, posi_train.shape[0], nega_num)
            train_samples = posi_train.tolist() + nega_train
            random.shuffle(train_samples)
            train_samples = np.array(train_samples)
            data_train = train_samples[:,:-1]
            y_train = train_samples[:,-1]
            print(y_train)
            #split train data into aas and sorfs
            sorfs_data_train = data_train[:,:sorfs_fea_num]
            aas_data_train = data_train[:,sorfs_fea_num:]
            
            # y_test_prob, sub_performance = ensemble_learning.ensem_learning_val(data_train, y_train, 0.3, data_test, aas_f_nef)
                        
            #aas
            aas_y_test_prob, aas_sub_performance = ensemble_learning.ensem_learning_val(aas_data_train, y_train, 0.3, aas_data_test, aas_f_nef, flag='aas')
            #sorfs
            sorfs_y_test_prob, sorfs_sub_performance = ensemble_learning.ensem_learning_val(sorfs_data_train, y_train, 0.3, sorfs_data_test, sorfs_f_nef, flag='sorfs')
            
            #ensemble
            y_test_prob = [(i+j)/2 for i, j in zip(aas_y_test_prob, sorfs_y_test_prob)]
            sub_performance = (aas_sub_performance+sorfs_sub_performance)/2
            
            y_test_probs.append(y_test_prob)
            sub_factors.append(sub_performance)
            tp, fp, tn, fn, BACC, MCC, f1_score, AUC, AUPR = score.calculate_performace(y_test_prob, y_test)
            print('\nSubset'+str(count+1)+'\ntp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\n  BACC = \t', BACC, '\n  MCC = \t', MCC, '\n  f1_score = \t', f1_score, '\n  AUC = \t', AUC, '\n  AUPR = \t', AUPR)

        sub_weights = ensemble_learning.get_subset_weights(sub_factors)
        print(sub_weights)
        final_prob = 0
        for i in range(len(y_test_probs)):
            final_prob += np.array(y_test_probs[i])*sub_weights[i]

        tp, fp, tn, fn, BACC, MCC, f1_score, AUC, AUPR = score.calculate_performace(final_prob, y_test)
        metric_list.append([tp, fp, tn, fn, BACC, MCC, f1_score, AUC, AUPR])

        print('\n------------------ Fold ', fold_index+1, '----------------------\ntp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\n  BACC = \t', BACC, '\n  MCC = \t', MCC, '\n  f1_score = \t', f1_score, '\n  AUC = \t', AUC, '\n  AUPR = \t', AUPR)
        fold_index += 1 

    fw = open('./Experimental results/CircPePred cv results.txt', 'a+')
    ave_tp, ave_fp, ave_tn, ave_fn, ave_BACC, ave_mcc, ave_f1_score, ave_auc, ave_aupr = score.get_average_metrics(metric_list)
    print('\n BACC = \t'+ str(ave_BACC)+ '\n MCC = \t'+str(ave_mcc)+'\n f1_score = \t'+str(ave_f1_score)+'\n AUC = \t'+ str(ave_auc) + '\n AUPR =\t'+str(ave_aupr)+'\n')
    fw.write('BACC\t'+ str(ave_BACC)+ '\tMCC\t'+str(ave_mcc)+'\tf1_score\t'+str(ave_f1_score)+'\tAUC\t'+ str(ave_auc) + '\tAUPR\t'+str(ave_aupr)+ "\tave_tp\t" + str(ave_tp) + "\tave_fp\t" + str(ave_fp) + "\tave_tn\t" + str(ave_tn) + "\tave_fn\t" + str(ave_fn) +'\n')
    fw.close()

CircPePred()