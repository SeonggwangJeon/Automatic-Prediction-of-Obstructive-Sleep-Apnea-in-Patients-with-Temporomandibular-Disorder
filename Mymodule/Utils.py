from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

today = datetime.now().strftime('%Y_%m_%d')


def calculate_metrics(conf_matrix):
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    
    return sensitivity, specificity, PPV, NPV


def decision_func(proba, threshold=0.5):
    prediction = []
    for p in proba:
        if p >= threshold:
            prediction.append(1)
        else:
            prediction.append(0)
    
    return np.asarray(prediction)



def padd_plot_confusion_matrix(figsize, xlabel,ylabel, conf_matrix, x_labels, y_labels, color, save = True, save_path= './Figures/test.png'):    
    pad_conf_matrix = conf_matrix
    
    total_num = int(conf_matrix.sum()/2)
    
    plt.figure(figsize=figsize, facecolor='white') # Change this as needed.
    plt.imshow(pad_conf_matrix, interpolation='nearest', cmap=color)
    plt.xlabel(xlabel, fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)
    plt.clim(-10, total_num)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)#, ticks = [0,100,200,300,400])
    cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()
    
    x_posits = np.arange(len(x_labels))
    y_posits = np.arange(len(y_labels))
    plt.xticks(x_posits, x_labels, fontsize = 15)
    plt.yticks(y_posits, y_labels, rotation=90, fontsize = 15)
    
    
    thresh = conf_matrix.max()/2.
    
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            plt.text(j,i, f'''{pad_conf_matrix[i,j]}\n({pad_conf_matrix[i,j]/pad_conf_matrix.sum()*100:.0f}%)''', fontsize=30, horizontalalignment="center", color="white")
    if save == True : plt.savefig(save_path)
    plt.show()
    

def bootstrap_auc(test_y, proba, n_bootstraps=1000):
    rng = np.random.RandomState(42)
    aucs = []

    for _ in range(n_bootstraps):
        indices = rng.choice(len(test_y), len(test_y), replace=True)
        bootstrapped_y = test_y[indices]
        bootstrapped_proba = proba[indices]
        if len(set(bootstrapped_y)) > 1:
            auc = metrics.roc_auc_score(bootstrapped_y, bootstrapped_proba)
            aucs.append(auc)

    lower_bound = np.percentile(aucs, 2.5)
    upper_bound = np.percentile(aucs, 97.5)

    return lower_bound, upper_bound

def plot_roc_curve(test_y, proba, title, size=(10,10), color='darkorange', save=True, save_path='./Figures/test', show = True, bootstrap = True):
    roc_auc = metrics.roc_auc_score(test_y, proba)
    fpr, tpr, ths = metrics.roc_curve(test_y, proba)

    best_acc = 0
    best_th = 0
    best_fpr = 0
    best_tpr = 0
    for th in ths:
        pred = decision_func(proba, th)
        acc = metrics.accuracy_score(test_y, pred)
        if acc > best_acc:
            best_acc = acc
            best_th = th
            best_tpr = tpr
            best_fpr = fpr

    # 부트스트래핑을 사용하여 AUC의 95% 신뢰 구간 계산
    
    if bootstrap == True:
        lower_ci, upper_ci = bootstrap_auc(test_y, proba)
        ci_label = '95% CI: {:.4f}-{:.4f}'.format(lower_ci, upper_ci)
    elif bootstrap == False:
        ci_label = ''
        #plt.text(0.5, 0.02, ci_label, ha='center', fontsize=12, transform=plt.gca().transAxes)
    
    if show == True : plt.figure(figsize=size, facecolor='white')
    lw = 2
    plt.plot(fpr, tpr, color=color,
             lw=lw, label=f'''AUC = {roc_auc:.4f}\n({ci_label})''' )

    plt.plot(best_fpr, best_tpr, color=color)

    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(title, fontsize=16)


    plt.legend(loc="lower right", fontsize=20)

    if save:
        plt.savefig(save_path)
        
    if show == True: plt.show()

    return roc_auc, best_th, best_acc



def plot_confusion_matrix_with_metrics_and_auc(figsize, xlabel, ylabel, test_y, proba, x_labels, y_labels, title, color, save=True, save_path='./Figures/test.png'):
    
    # Plot AUC Curve
    
    plt.figure(figsize=figsize, facecolor='white')
    
    plt.subplot(1, 3, 1)
    roc_auc, best_th, best_acc = plot_roc_curve(test_y, proba, title, size=(10, 10), color='blue', save=False, show = False)
    
    thres = best_th
    
    #print(best_th)
    if len(set(proba)) <= 2 :
        pred_y = proba
    else : 
        pred_y = decision_func(proba, threshold=thres)
          
    conf_matrix = confusion_matrix(test_y, pred_y)
    pad_conf_matrix = conf_matrix

    total_num = int(conf_matrix.sum() / 2)

    # Plot Confusion Matrix
    plt.subplot(1, 3, 2)
    plt.imshow(pad_conf_matrix, interpolation='nearest', cmap=color)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.clim(-10, total_num)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()

    x_posits = np.arange(len(x_labels))
    y_posits = np.arange(len(y_labels))
    plt.xticks(x_posits, x_labels, fontsize=15)
    plt.yticks(y_posits, y_labels, rotation=90, fontsize=15)

    thresh = conf_matrix.max() / 2.

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            plt.text(j, i, f'{pad_conf_matrix[i, j]}\n({pad_conf_matrix[i, j] / pad_conf_matrix.sum() * 100:.0f}%)', fontsize=30, horizontalalignment="center", color="white")

    # Add Metrics as Text
    plt.subplot(1, 3, 3)
    plt.axis('off')
    sensitivity, specificity, PPV, NPV = calculate_metrics(conf_matrix)
    metrics_labels = ['Metrics', 'Value', 'Sensitivity', f'{sensitivity:.2f}', 'Specificity', f'{specificity:.2f}', 'PPV', f'{PPV:.2f}', 'NPV', f'{NPV:.2f}']
    table_data = [[metrics_labels[i], metrics_labels[i + 1]] for i in range(0, len(metrics_labels), 2)]
    table = plt.table(cellText=table_data, cellLoc='center', colLabels=None, cellColours=[['lightgray']*2]*5, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(0.75, 1.5)

    if save: plt.savefig(save_path)
        
    plt.show()
    
    

def z_score_normalize(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - np.mean(lst)) / np.std(lst)
        normalized.append(normalized_num)
    return normalized